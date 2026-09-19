USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Reconciles every upstream batch feed registered for today
               against its expected row count, queuing a retry for feeds
               that are still short one time before flagging them as
               failed, upserts a reconciliation outcome into the
               reconciliation summary table, and transitions each feed's
               status with a full audit trail.
 EXEC PRO.Batch_Job_Reconciliation @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.Batch_Job_Reconciliation
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        DECLARE @ProcessDate DATE = (SELECT [Date] FROM SysDayMatrix WHERE TimeKey = @TimeKey)
        DECLARE @MonthEndDate DATE = EOMONTH(@ProcessDate)

        IF OBJECT_ID('tempdb..#ReconciliationResults') IS NOT NULL
            DROP TABLE #ReconciliationResults

        CREATE TABLE #ReconciliationResults
        (
            FeedName     VARCHAR(50),
            Outcome      VARCHAR(20),
            ShortfallPct DECIMAL(9,4),
            SeverityTier VARCHAR(10),
            ReconciledOn DATE
        )

        -- Rule 1: NULL check - a feed with no expected count on record
        -- cannot be reconciled at all; sequential IF/ELSE - a feed
        -- matching or exceeding its expected row count reconciles
        -- cleanly; nested IF/ELSE - a short feed beyond tolerance on the
        -- last day of the month fails immediately regardless of retry
        -- count, otherwise a short feed not yet retried is queued for
        -- one retry - all expressed as a multi-branch CASE, set-based
        INSERT INTO #ReconciliationResults (FeedName, Outcome, ShortfallPct, SeverityTier, ReconciledOn)
        SELECT
            FeedName,
            CASE
                WHEN ExpectedRowCount IS NULL OR ExpectedRowCount = 0 THEN 'NOT_APPLICABLE'
                WHEN ActualRowCount >= ExpectedRowCount THEN 'RECONCILED'
                WHEN @ProcessDate = @MonthEndDate
                     AND (CAST(ExpectedRowCount - ActualRowCount AS DECIMAL(18,4)) / ExpectedRowCount) * 100 > 10
                    THEN 'FAILED'
                WHEN ISNULL(RetryCount, 0) = 0 THEN 'RETRY_QUEUED'
                ELSE 'FAILED'
            END,
            CASE
                WHEN ExpectedRowCount IS NULL OR ExpectedRowCount = 0 THEN NULL
                WHEN ActualRowCount >= ExpectedRowCount THEN 0
                ELSE (CAST(ExpectedRowCount - ActualRowCount AS DECIMAL(18,4)) / ExpectedRowCount) * 100
            END,
            NULL,
            @ProcessDate
        FROM PRO.BatchFeedRegistry
        WHERE FeedDate = @ProcessDate

        -- Rule 2: mark the registry row for another attempt for every
        -- feed staged as RETRY_QUEUED above
        UPDATE PRO.BatchFeedRegistry
        SET RetryCount = 1
        WHERE FeedDate = @ProcessDate
          AND FeedName IN (SELECT FeedName FROM #ReconciliationResults WHERE Outcome = 'RETRY_QUEUED')

        -- Rule 3: multi-branch CASE - derive a severity tier for every
        -- outcome staged above, read back from the staging table
        UPDATE R
        SET R.SeverityTier = (
                CASE
                    WHEN R.Outcome = 'RECONCILED' THEN 'NONE'
                    WHEN R.Outcome = 'NOT_APPLICABLE' THEN 'NONE'
                    WHEN R.Outcome = 'RETRY_QUEUED' THEN 'WATCH'
                    WHEN R.Outcome = 'FAILED' AND R.ShortfallPct > 25 THEN 'CRITICAL'
                    WHEN R.Outcome = 'FAILED' THEN 'HIGH'
                    ELSE 'UNKNOWN'
                END
            )
        FROM #ReconciliationResults R

        -- Rule 4: upsert every outcome computed above into the
        -- reconciliation summary table - refresh feeds already tracked,
        -- add feeds seen for the first time this run
        MERGE PRO.BatchReconciliationSummary AS Target
        USING #ReconciliationResults AS Source
        ON Target.FeedName = Source.FeedName
        WHEN MATCHED THEN
            UPDATE SET
                Target.Outcome = Source.Outcome,
                Target.ShortfallPct = Source.ShortfallPct,
                Target.SeverityTier = Source.SeverityTier,
                Target.LastReconciledDate = Source.ReconciledOn
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (FeedName, Outcome, ShortfallPct, SeverityTier, FirstReconciledDate, LastReconciledDate)
            VALUES (Source.FeedName, Source.Outcome, Source.ShortfallPct, Source.SeverityTier, Source.ReconciledOn, Source.ReconciledOn);

        -- Rule 5: date comparison/range - feeds that have not reconciled
        -- cleanly in the last 14 days are escalated regardless of today's
        -- outcome, cross-referencing the summary table just upserted above
        UPDATE T
        SET T.SeverityTier = 'CRITICAL'
        FROM PRO.BatchReconciliationSummary T
        WHERE T.Outcome <> 'RECONCILED'
          AND T.FirstReconciledDate <= DATEADD(DAY, -14, @ProcessDate)

        -- Rule 6: persist every outcome computed above into the log,
        -- read back from the staging table populated above
        INSERT INTO PRO.BatchReconciliationLog (FeedName, Outcome, ReconciledOn)
        SELECT R.FeedName, R.Outcome, R.ReconciledOn
        FROM #ReconciliationResults R

        -- Rule 7: status transition and audit - failed feeds are
        -- escalated to the collections-style queue for operations
        INSERT INTO PRO.CollectionsQueue (AccountId, EscalationDate, Reason)
        SELECT FeedName, ReconciledOn,
               CASE WHEN SeverityTier = 'CRITICAL' THEN 'FEED_RECONCILIATION_CRITICAL' ELSE 'FEED_RECONCILIATION_FAILED' END
        FROM #ReconciliationResults
        WHERE Outcome = 'FAILED'

        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'Y', ERRORDATE = NULL, ERRORDESCRIPTION = NULL, COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Batch_Job_Reconciliation'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'N', ERRORDATE = GETDATE(), ERRORDESCRIPTION = ERROR_MESSAGE(), COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Batch_Job_Reconciliation'
    END CATCH
    SET NOCOUNT OFF
END
GO
