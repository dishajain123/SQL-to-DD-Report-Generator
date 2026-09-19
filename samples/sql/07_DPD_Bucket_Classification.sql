USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Classifies each loan account into a days-past-due (DPD)
               bucket, calculates bucket-wise penal interest, stages and
               upserts the classification into the DPD history table,
               queues notifications for newly-worsened accounts, and
               logs every status transition for audit.
 EXEC PRO.DPD_Bucket_Classification @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.DPD_Bucket_Classification
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        DECLARE @ProcessDate DATE = (SELECT [Date] FROM SysDayMatrix WHERE TimeKey = @TimeKey)
        DECLARE @GraceWindowStart DATE = DATEADD(DAY, -3, @ProcessDate)

        -- Rule 1: derive days past due from the last payment date
        UPDATE A
        SET A.DpdDays = DATEDIFF(DAY, A.LastPaymentDueDate, @ProcessDate)
        FROM PRO.LoanAccountCal A
        WHERE A.LastPaymentDueDate IS NOT NULL
          AND A.LastPaymentDueDate <= @ProcessDate

        -- Rule 2: accounts with no due date on record cannot be aged
        UPDATE A
        SET A.DpdDays = 0,
            A.DpdBucket = 'NOT_APPLICABLE'
        FROM PRO.LoanAccountCal A
        WHERE A.LastPaymentDueDate IS NULL

        -- Rule 3: multi-branch bucket classification by DPD range
        UPDATE A
        SET A.DpdBucket = (
                CASE
                    WHEN A.DpdDays IS NULL THEN 'NOT_APPLICABLE'
                    WHEN A.DpdDays = 0 THEN 'CURRENT'
                    WHEN A.DpdDays BETWEEN 1 AND 30 THEN 'BUCKET_1_30'
                    WHEN A.DpdDays BETWEEN 31 AND 60 THEN 'BUCKET_31_60'
                    WHEN A.DpdDays BETWEEN 61 AND 90 THEN 'BUCKET_61_90'
                    ELSE 'BUCKET_90_PLUS'
                END
            )
        FROM PRO.LoanAccountCal A
        WHERE A.DpdDays IS NOT NULL

        -- Rule 4: penal interest applies only once an account is overdue,
        -- and scales with how late the account is
        UPDATE A
        SET A.PenalInterestAmount = (
                CASE
                    WHEN A.DpdBucket = 'BUCKET_1_30' THEN (A.OutstandingBalance * 0.02) / 365 * A.DpdDays
                    WHEN A.DpdBucket = 'BUCKET_31_60' THEN (A.OutstandingBalance * 0.03) / 365 * A.DpdDays
                    WHEN A.DpdBucket IN ('BUCKET_61_90', 'BUCKET_90_PLUS') THEN (A.OutstandingBalance * 0.04) / 365 * A.DpdDays
                    ELSE 0
                END
            )
        FROM PRO.LoanAccountCal A

        -- Rule 5: sequential IF/ELSE - accounts within the 3-day grace
        -- window skip escalation entirely this cycle, everything else
        -- worsened relative to the prior run is flagged
        IF EXISTS (SELECT 1 FROM PRO.LoanAccountCal WHERE LastPaymentDueDate >= @GraceWindowStart)
        BEGIN
            UPDATE A
            SET A.BucketWorsened = 'N',
                A.GracePeriodApplied = 'Y'
            FROM PRO.LoanAccountCal A
            WHERE A.LastPaymentDueDate >= @GraceWindowStart
        END
        ELSE IF EXISTS (SELECT 1 FROM PRO.LoanAccountCal WHERE PrevDpdBucket IS NOT NULL)
        BEGIN
            UPDATE A
            SET A.BucketWorsened = 'Y'
            FROM PRO.LoanAccountCal A
            WHERE A.PrevDpdBucket IS NOT NULL
              AND A.DpdBucket <> A.PrevDpdBucket
              AND A.DpdDays > ISNULL(A.PrevDpdDays, 0)
        END
        ELSE
        BEGIN
            UPDATE A
            SET A.BucketWorsened = 'N'
            FROM PRO.LoanAccountCal A
        END

        IF OBJECT_ID('tempdb..#DpdStaging') IS NOT NULL
            DROP TABLE #DpdStaging

        CREATE TABLE #DpdStaging
        (
            AccountId       VARCHAR(20),
            DpdBucket       VARCHAR(20),
            FacilityType    VARCHAR(10),
            AdjustedPenalty DECIMAL(18,2)
        )

        -- Rule 6: conditional INSERT - only stage accounts that actually
        -- worsened this cycle, not the full population
        INSERT INTO #DpdStaging (AccountId, DpdBucket, FacilityType, AdjustedPenalty)
        SELECT A.AccountId, A.DpdBucket, A.FacilityType, A.PenalInterestAmount
        FROM PRO.LoanAccountCal A
        WHERE A.BucketWorsened = 'Y'

        -- Rule 7: derived-assignment UPDATE reading back the staging rows
        -- written above - facility type nudges the penalty up or down
        UPDATE S
        SET S.AdjustedPenalty = (
                CASE
                    WHEN S.FacilityType IN ('CC', 'OD') THEN S.AdjustedPenalty * 1.10
                    WHEN S.FacilityType IN ('TL', 'DL') THEN S.AdjustedPenalty * 1.05
                    ELSE S.AdjustedPenalty
                END
            )
        FROM #DpdStaging S
        WHERE S.AdjustedPenalty IS NOT NULL

        -- Rule 8: upsert the staged, adjusted rows into the DPD history
        -- table - refresh accounts already tracked, add new ones
        MERGE PRO.DpdBucketHistory AS Target
        USING #DpdStaging AS Source
        ON Target.AccountId = Source.AccountId
        WHEN MATCHED THEN
            UPDATE SET
                Target.DpdBucket = Source.DpdBucket,
                Target.AdjustedPenalty = Source.AdjustedPenalty,
                Target.LastUpdatedDate = @ProcessDate
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (AccountId, DpdBucket, AdjustedPenalty, FirstFlaggedDate, LastUpdatedDate)
            VALUES (Source.AccountId, Source.DpdBucket, Source.AdjustedPenalty, @ProcessDate, @ProcessDate);

        -- Rule 9: queue a notification for every newly-worsened account -
        -- nested IF/ELSE (severity first, then facility type within the
        -- severe branch) expressed as a multi-branch CASE, set-based
        INSERT INTO PRO.CollectionsQueue (AccountId, EscalationDate, Reason)
        SELECT AccountId, @ProcessDate,
               CASE
                   WHEN DpdBucket IN ('BUCKET_61_90', 'BUCKET_90_PLUS') AND FacilityType IN ('CC', 'OD')
                       THEN 'SEVERE_DPD_CASH_CREDIT'
                   WHEN DpdBucket IN ('BUCKET_61_90', 'BUCKET_90_PLUS')
                       THEN 'SEVERE_DPD_OTHER_FACILITY'
                   ELSE 'EARLY_DPD_WORSENED'
               END
        FROM PRO.LoanAccountCal
        WHERE BucketWorsened = 'Y'

        -- Rule 10: status-transition audit row for every bucket change
        -- applied this run, cross-referencing the history table above
        INSERT INTO PRO.DpdBucketAuditLog (AccountId, TransitionDate, NewBucket)
        SELECT H.AccountId, @ProcessDate, H.DpdBucket
        FROM PRO.DpdBucketHistory H
        WHERE H.LastUpdatedDate = @ProcessDate

        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'Y', ERRORDATE = NULL, ERRORDESCRIPTION = NULL, COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'DPD_Bucket_Classification'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'N', ERRORDATE = GETDATE(), ERRORDESCRIPTION = ERROR_MESSAGE(), COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'DPD_Bucket_Classification'
    END CATCH
    SET NOCOUNT OFF
END
GO
