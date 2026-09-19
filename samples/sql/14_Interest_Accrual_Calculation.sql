USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Accrues daily interest for each active loan account since
               its last accrual date, applies a concessional rate for
               promotional accounts, stages and upserts the accrual into
               the interest ledger, walks large-accrual accounts to log
               a review case, and rolls the accrued amount into balance.
 EXEC PRO.Interest_Accrual_Calculation @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.Interest_Accrual_Calculation
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        DECLARE @ProcessDate DATE = (SELECT [Date] FROM SysDayMatrix WHERE TimeKey = @TimeKey)
        DECLARE @PromoWindowStart DATE = DATEADD(DAY, -90, @ProcessDate)

        -- Rule 1: number of days to accrue since the last accrual run;
        -- accounts never accrued before default to a single day
        UPDATE A
        SET A.DaysSinceLastAccrual = (
                CASE
                    WHEN A.LastAccrualDate IS NULL THEN 1
                    ELSE DATEDIFF(DAY, A.LastAccrualDate, @ProcessDate)
                END
            )
        FROM PRO.LoanAccountCal A
        WHERE A.AccountStatus = 'ACTIVE'

        -- Rule 2: sequential IF/ELSE - accounts opened within the
        -- promotional window and flagged promotional get a concessional
        -- rate; everything else uses its base rate
        IF EXISTS (SELECT 1 FROM PRO.LoanAccountCal WHERE PromotionalFlag = 'Y' AND AccountOpenDate >= @PromoWindowStart)
        BEGIN
            UPDATE A
            SET A.EffectiveInterestRate = A.BaseInterestRate - 0.02
            FROM PRO.LoanAccountCal A
            WHERE A.AccountStatus = 'ACTIVE'
              AND A.PromotionalFlag = 'Y'
              AND A.AccountOpenDate >= @PromoWindowStart

            UPDATE A
            SET A.EffectiveInterestRate = A.BaseInterestRate
            FROM PRO.LoanAccountCal A
            WHERE A.AccountStatus = 'ACTIVE'
              AND (A.PromotionalFlag = 'N' OR A.PromotionalFlag IS NULL
                   OR A.AccountOpenDate < @PromoWindowStart)
        END
        ELSE
        BEGIN
            UPDATE A
            SET A.EffectiveInterestRate = A.BaseInterestRate
            FROM PRO.LoanAccountCal A
            WHERE A.AccountStatus = 'ACTIVE'
        END

        IF OBJECT_ID('tempdb..#AccrualStaging') IS NOT NULL
            DROP TABLE #AccrualStaging

        CREATE TABLE #AccrualStaging
        (
            AccountId       VARCHAR(20),
            AccruedInterestAmount DECIMAL(18,2),
            AccrualTier     VARCHAR(10)
        )

        -- Rule 3: conditional INSERT - compute and stage the accrued
        -- interest only for active accounts with a known balance
        INSERT INTO #AccrualStaging (AccountId, AccruedInterestAmount, AccrualTier)
        SELECT A.AccountId,
               (A.OutstandingBalance * A.EffectiveInterestRate / 365) * A.DaysSinceLastAccrual,
               CASE
                   WHEN (A.OutstandingBalance * A.EffectiveInterestRate / 365) * A.DaysSinceLastAccrual > 5000 THEN 'LARGE'
                   WHEN (A.OutstandingBalance * A.EffectiveInterestRate / 365) * A.DaysSinceLastAccrual > 1000 THEN 'MEDIUM'
                   ELSE 'SMALL'
               END
        FROM PRO.LoanAccountCal A
        WHERE A.AccountStatus = 'ACTIVE'
          AND A.OutstandingBalance IS NOT NULL

        -- Rule 4: upsert the staged accrual into the interest ledger -
        -- refresh accounts already tracked, add accounts accrued for the
        -- first time this run
        MERGE PRO.InterestAccrualLedger AS Target
        USING #AccrualStaging AS Source
        ON Target.AccountId = Source.AccountId
        WHEN MATCHED THEN
            UPDATE SET
                Target.AccruedInterestAmount = Source.AccruedInterestAmount,
                Target.AccrualTier = Source.AccrualTier,
                Target.LastAccrualDate = @ProcessDate
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (AccountId, AccruedInterestAmount, AccrualTier, FirstAccrualDate, LastAccrualDate)
            VALUES (Source.AccountId, Source.AccruedInterestAmount, Source.AccrualTier, @ProcessDate, @ProcessDate);

        -- Rule 5: roll the accrued interest into the outstanding balance
        -- and stamp the accrual date, read back from the staging table
        UPDATE A
        SET A.OutstandingBalance = A.OutstandingBalance + ISNULL(S.AccruedInterestAmount, 0),
            A.LastAccrualDate = @ProcessDate
        FROM PRO.LoanAccountCal A
        INNER JOIN #AccrualStaging S ON S.AccountId = A.AccountId

        -- Rule 6: log a review case for every large-accrual account, read
        -- back from the staging table populated above
        INSERT INTO PRO.CollateralReview (AccountId, ReviewDate, ShortfallAmount)
        SELECT AccountId, @ProcessDate, AccruedInterestAmount
        FROM #AccrualStaging
        WHERE AccrualTier = 'LARGE'

        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'Y', ERRORDATE = NULL, ERRORDESCRIPTION = NULL, COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Interest_Accrual_Calculation'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'N', ERRORDATE = GETDATE(), ERRORDESCRIPTION = ERROR_MESSAGE(), COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Interest_Accrual_Calculation'
    END CATCH
    SET NOCOUNT OFF
END
GO
