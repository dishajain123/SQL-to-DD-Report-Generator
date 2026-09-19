USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Stages the latest collateral valuations, derives a
               shortfall amount against each secured loan's outstanding
               balance, upserts the position into the collateral summary
               table, walks shortfall accounts to open review cases, and
               flags stale valuations separately.
 EXEC PRO.Collateral_Valuation_Staging @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.Collateral_Valuation_Staging
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        DECLARE @ProcessDate DATE = (SELECT [Date] FROM SysDayMatrix WHERE TimeKey = @TimeKey)
        DECLARE @StaleValuationCutoff DATE = DATEADD(MONTH, -12, @ProcessDate)

        IF OBJECT_ID('tempdb..#CollateralStaging') IS NOT NULL
            DROP TABLE #CollateralStaging

        CREATE TABLE #CollateralStaging
        (
            AccountId          VARCHAR(20),
            CollateralValue    DECIMAL(18,2),
            OutstandingBalance DECIMAL(18,2),
            ShortfallAmount    DECIMAL(18,2),
            ReviewPriority     VARCHAR(10)
        )

        -- Rule 1: conditional INSERT - stage only secured accounts that
        -- have a recorded, non-stale collateral valuation this cycle
        INSERT INTO #CollateralStaging (AccountId, CollateralValue, OutstandingBalance, ShortfallAmount, ReviewPriority)
        SELECT A.AccountId,
               V.LatestValuationAmount,
               A.OutstandingBalance,
               NULL,
               NULL
        FROM PRO.LoanAccountCal A
        INNER JOIN PRO.CollateralValuation V ON V.AccountId = A.AccountId
        WHERE A.SecuredFlag = 'Y'
          AND V.ValuationDate >= @StaleValuationCutoff

        -- Rule 2: derive the shortfall for staged rows, read back from the
        -- staging table populated above
        UPDATE S
        SET S.ShortfallAmount = S.OutstandingBalance - S.CollateralValue
        FROM #CollateralStaging S
        WHERE S.OutstandingBalance > S.CollateralValue

        -- Rule 3: rows with adequate collateral have no shortfall
        UPDATE S
        SET S.ShortfallAmount = 0
        FROM #CollateralStaging S
        WHERE S.OutstandingBalance <= S.CollateralValue

        -- Rule 4: sequential IF/ELSE, then multi-branch CASE - review
        -- priority depends on both how close to quarter-end we are and
        -- the size of the shortfall
        IF DATEPART(MONTH, @ProcessDate) IN (3, 6, 9, 12)
        BEGIN
            UPDATE S
            SET S.ReviewPriority = (
                    CASE
                        WHEN S.ShortfallAmount IS NULL THEN 'NONE'
                        WHEN S.ShortfallAmount > 500000 THEN 'URGENT'
                        WHEN S.ShortfallAmount > 100000 THEN 'HIGH'
                        WHEN S.ShortfallAmount > 0 THEN 'STANDARD'
                        ELSE 'NONE'
                    END
                )
            FROM #CollateralStaging S
        END
        ELSE
        BEGIN
            UPDATE S
            SET S.ReviewPriority = (
                    CASE
                        WHEN S.ShortfallAmount > 500000 THEN 'HIGH'
                        WHEN S.ShortfallAmount > 0 THEN 'STANDARD'
                        ELSE 'NONE'
                    END
                )
            FROM #CollateralStaging S
        END

        -- Rule 5: upsert the staged position into the collateral summary
        -- table - refresh accounts already tracked, add new ones
        MERGE PRO.CollateralPositionSummary AS Target
        USING #CollateralStaging AS Source
        ON Target.AccountId = Source.AccountId
        WHEN MATCHED THEN
            UPDATE SET
                Target.CollateralValue = Source.CollateralValue,
                Target.ShortfallAmount = Source.ShortfallAmount,
                Target.ReviewPriority = Source.ReviewPriority,
                Target.LastUpdatedDate = @ProcessDate
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (AccountId, CollateralValue, ShortfallAmount, ReviewPriority, FirstSeenDate, LastUpdatedDate)
            VALUES (Source.AccountId, Source.CollateralValue, Source.ShortfallAmount, Source.ReviewPriority, @ProcessDate, @ProcessDate);

        -- Rule 6: open a review case for every shortfall account, read
        -- back from the staging table populated above
        INSERT INTO PRO.CollateralReview (AccountId, ReviewDate, ShortfallAmount)
        SELECT AccountId, @ProcessDate, ShortfallAmount
        FROM #CollateralStaging
        WHERE ShortfallAmount > 0

        -- Rule 7: secured accounts with no valuation in the last 12 months
        -- are flagged separately as stale, independent of shortfall - a
        -- NULL/NOT EXISTS driven status update
        UPDATE A
        SET A.CollateralStale = 'Y'
        FROM PRO.LoanAccountCal A
        WHERE A.SecuredFlag = 'Y'
          AND NOT EXISTS (
              SELECT 1 FROM PRO.CollateralValuation V
              WHERE V.AccountId = A.AccountId
                AND V.ValuationDate >= @StaleValuationCutoff
          )

        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'Y', ERRORDATE = NULL, ERRORDESCRIPTION = NULL, COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Collateral_Valuation_Staging'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'N', ERRORDATE = GETDATE(), ERRORDESCRIPTION = ERROR_MESSAGE(), COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Collateral_Valuation_Staging'
    END CATCH
    SET NOCOUNT OFF
END
GO
