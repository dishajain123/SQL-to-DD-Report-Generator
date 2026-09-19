USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Recomputes each account's provision coverage ratio, stages
               and upserts the result into the provision summary table,
               walks accounts that fell below threshold to log a review
               reason, and records a status-transition audit row for
               every coverage change applied.
 EXEC PRO.Provision_Coverage_Merge @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.Provision_Coverage_Merge
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        DECLARE @ProcessDate DATE = (SELECT [Date] FROM SysDayMatrix WHERE TimeKey = @TimeKey)
        DECLARE @QuarterStartDate DATE = DATEADD(MONTH, -3, @ProcessDate)

        IF OBJECT_ID('tempdb..#ProvisionCoverage') IS NOT NULL
            DROP TABLE #ProvisionCoverage

        CREATE TABLE #ProvisionCoverage
        (
            AccountId        VARCHAR(20),
            OutstandingBalance DECIMAL(18,2),
            ProvisionAmount  DECIMAL(18,2),
            CoverageRatio    DECIMAL(9,4),
            ReviewReason     VARCHAR(40)
        )

        -- Rule 1: conditional INSERT - coverage ratio is only staged for
        -- accounts carrying a positive outstanding balance
        INSERT INTO #ProvisionCoverage (AccountId, OutstandingBalance, ProvisionAmount, CoverageRatio, ReviewReason)
        SELECT A.AccountId,
               A.OutstandingBalance,
               A.ProvisionAmount,
               CASE
                   WHEN A.OutstandingBalance > 0 THEN A.ProvisionAmount / A.OutstandingBalance
                   ELSE NULL
               END,
               NULL
        FROM PRO.LoanAccountCal A
        WHERE A.OutstandingBalance IS NOT NULL

        -- Rule 2: derived-assignment UPDATE reading back the staging table
        -- above - a multi-branch review reason based on coverage and age
        UPDATE S
        SET S.ReviewReason = (
                CASE
                    WHEN S.CoverageRatio IS NULL THEN 'NO_BALANCE'
                    WHEN S.CoverageRatio < 0.25 THEN 'SEVERE_UNDERCOVER'
                    WHEN S.CoverageRatio < 0.5 THEN 'MODERATE_UNDERCOVER'
                    WHEN S.CoverageRatio >= 0.5 AND S.CoverageRatio < 1 THEN 'ADEQUATE'
                    ELSE 'FULLY_COVERED'
                END
            )
        FROM #ProvisionCoverage S

        -- Rule 3: sequential IF/ELSE - during the last month of the
        -- quarter, apply a stricter threshold for the below-threshold flag
        IF @ProcessDate >= DATEADD(MONTH, 1, @QuarterStartDate)
        BEGIN
            UPDATE S SET S.ReviewReason = S.ReviewReason + '_QUARTER_END'
            FROM #ProvisionCoverage S
            WHERE S.CoverageRatio IS NOT NULL AND S.CoverageRatio < 0.6
        END
        ELSE
        BEGIN
            UPDATE S SET S.ReviewReason = S.ReviewReason
            FROM #ProvisionCoverage S
            WHERE S.CoverageRatio IS NOT NULL AND S.CoverageRatio < 0.5
        END

        -- Rule 4: upsert the computed coverage into the summary table -
        -- update accounts already tracked, insert accounts seen for the
        -- first time this run
        MERGE PRO.ProvisionCoverageSummary AS Target
        USING #ProvisionCoverage AS Source
        ON Target.AccountId = Source.AccountId
        WHEN MATCHED THEN
            UPDATE SET
                Target.OutstandingBalance = Source.OutstandingBalance,
                Target.ProvisionAmount = Source.ProvisionAmount,
                Target.CoverageRatio = Source.CoverageRatio,
                Target.LastUpdatedDate = @ProcessDate
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (AccountId, OutstandingBalance, ProvisionAmount, CoverageRatio, FirstSeenDate, LastUpdatedDate)
            VALUES (Source.AccountId, Source.OutstandingBalance, Source.ProvisionAmount, Source.CoverageRatio, @ProcessDate, @ProcessDate);

        -- Rule 5: status transition - accounts that fell below the
        -- minimum coverage threshold are flagged for the risk committee
        UPDATE T
        SET T.BelowThresholdFlag = 'Y'
        FROM PRO.ProvisionCoverageSummary T
        WHERE T.CoverageRatio IS NOT NULL
          AND T.CoverageRatio < 0.5
          AND T.LastUpdatedDate = @ProcessDate

        -- Rule 6: accounts at or above the threshold are cleared of any
        -- earlier flag
        UPDATE T
        SET T.BelowThresholdFlag = 'N'
        FROM PRO.ProvisionCoverageSummary T
        WHERE T.CoverageRatio >= 0.5
          AND T.LastUpdatedDate = @ProcessDate

        -- Rule 7: record a review case for every account newly below
        -- threshold - the escalation queue depends on how severe the
        -- shortfall is, expressed as a multi-branch CASE, set-based
        INSERT INTO PRO.CollectionsQueue (AccountId, EscalationDate, Reason)
        SELECT AccountId, @ProcessDate,
               CASE WHEN CoverageRatio < 0.25 THEN 'SEVERE_PROVISION_SHORTFALL' ELSE 'MODERATE_PROVISION_SHORTFALL' END
        FROM #ProvisionCoverage
        WHERE ReviewReason LIKE '%UNDERCOVER%'
          AND CoverageRatio IS NOT NULL

        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'Y', ERRORDATE = NULL, ERRORDESCRIPTION = NULL, COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Provision_Coverage_Merge'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'N', ERRORDATE = GETDATE(), ERRORDESCRIPTION = ERROR_MESSAGE(), COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Provision_Coverage_Merge'
    END CATCH
    SET NOCOUNT OFF
END
GO
