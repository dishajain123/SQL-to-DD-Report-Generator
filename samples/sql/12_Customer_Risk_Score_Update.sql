USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Recomputes each customer's composite risk score from
               overdue history, utilization, and prior default count,
               stages and upserts the result into the risk score history
               table, walks newly-scored customers to log an audit row,
               and derives a recommended credit limit adjustment.
 EXEC PRO.Customer_Risk_Score_Update @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.Customer_Risk_Score_Update
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        DECLARE @ProcessDate DATE = (SELECT [Date] FROM SysDayMatrix WHERE TimeKey = @TimeKey)
        DECLARE @ReviewCycleStart DATE = DATEADD(MONTH, -1, @ProcessDate)

        -- Rule 1: utilization ratio is only computed where a credit limit
        -- is on record and positive
        UPDATE C
        SET C.UtilizationRatio = C.CurrentBalance / C.CreditLimit
        FROM PRO.CustomerRiskProfile C
        WHERE C.CreditLimit IS NOT NULL
          AND C.CreditLimit > 0

        -- Rule 2: composite score combines overdue days, utilization, and
        -- prior defaults into a single weighted figure
        UPDATE C
        SET C.RiskScore =
            (ISNULL(C.OverdueDays, 0) * 0.5)
            + (ISNULL(C.UtilizationRatio, 0) * 100 * 0.3)
            + (ISNULL(C.PriorDefaultCount, 0) * 20 * 0.2)
        FROM PRO.CustomerRiskProfile C

        -- Rule 3: multi-branch risk tier derived from the composite score
        UPDATE C
        SET C.RiskTier = (
                CASE
                    WHEN C.RiskScore < 20 THEN 'LOW'
                    WHEN C.RiskScore < 50 THEN 'MODERATE'
                    WHEN C.RiskScore < 80 THEN 'HIGH'
                    ELSE 'SEVERE'
                END
            )
        FROM PRO.CustomerRiskProfile C

        -- Rule 4: sequential IF/ELSE, nested inside - within the monthly
        -- review cycle, the limit adjustment also considers whether the
        -- customer had any prior default at all
        IF @ProcessDate >= @ReviewCycleStart
        BEGIN
            UPDATE C
            SET C.RecommendedLimitAdjustmentPct =
                CASE
                    WHEN C.RiskTier = 'LOW' THEN
                        CASE WHEN ISNULL(C.PriorDefaultCount, 0) = 0 THEN 10 ELSE 0 END
                    WHEN C.RiskTier = 'MODERATE' THEN 0
                    WHEN C.RiskTier = 'HIGH' THEN -15
                    WHEN C.RiskTier = 'SEVERE' THEN -40
                    ELSE 0
                END
            FROM PRO.CustomerRiskProfile C
        END
        ELSE
        BEGIN
            UPDATE C
            SET C.RecommendedLimitAdjustmentPct = 0
            FROM PRO.CustomerRiskProfile C
        END

        IF OBJECT_ID('tempdb..#RiskScoreStaging') IS NOT NULL
            DROP TABLE #RiskScoreStaging

        CREATE TABLE #RiskScoreStaging
        (
            CustomerId  VARCHAR(20),
            RiskScore   DECIMAL(9,2),
            RiskTier    VARCHAR(10)
        )

        -- Rule 5: conditional INSERT - only customers with a computed
        -- score this cycle are staged
        INSERT INTO #RiskScoreStaging (CustomerId, RiskScore, RiskTier)
        SELECT CustomerId, RiskScore, RiskTier
        FROM PRO.CustomerRiskProfile
        WHERE RiskScore IS NOT NULL

        -- Rule 6: upsert the staged score into the history table -
        -- refresh customers already tracked, add customers scored for
        -- the first time this run
        MERGE PRO.RiskScoreHistory AS Target
        USING #RiskScoreStaging AS Source
        ON Target.CustomerId = Source.CustomerId
        WHEN MATCHED THEN
            UPDATE SET
                Target.RiskScore = Source.RiskScore,
                Target.RiskTier = Source.RiskTier,
                Target.LastScoredDate = @ProcessDate
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (CustomerId, RiskScore, RiskTier, FirstScoredDate, LastScoredDate)
            VALUES (Source.CustomerId, Source.RiskScore, Source.RiskTier, @ProcessDate, @ProcessDate);

        -- Rule 7: log an audit row for every customer newly scored this
        -- run (no prior score on record)
        INSERT INTO PRO.RiskScoreAuditLog (CustomerId, ScoreDate, RiskScore, RiskTier)
        SELECT CustomerId, @ProcessDate, RiskScore, RiskTier
        FROM PRO.CustomerRiskProfile
        WHERE PrevRiskScore IS NULL
          AND RiskScore IS NOT NULL

        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'Y', ERRORDATE = NULL, ERRORDESCRIPTION = NULL, COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Customer_Risk_Score_Update'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'N', ERRORDATE = GETDATE(), ERRORDESCRIPTION = ERROR_MESSAGE(), COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Customer_Risk_Score_Update'
    END CATCH
    SET NOCOUNT OFF
END
GO
