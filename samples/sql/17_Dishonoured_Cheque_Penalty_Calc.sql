USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Applies a penalty for each dishonoured cheque recorded
               today, scaling the penalty by repeat-dishonour count,
               stages and upserts the penalty into the cheque penalty
               ledger, walks accounts crossing the repeat-dishonour
               threshold to suspend cheque-book privileges, and audits
               every suspension applied.
 EXEC PRO.Dishonoured_Cheque_Penalty_Calc @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.Dishonoured_Cheque_Penalty_Calc
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        DECLARE @ProcessDate DATE = (SELECT [Date] FROM SysDayMatrix WHERE TimeKey = @TimeKey)
        DECLARE @LookbackWindowStart DATE = DATEADD(MONTH, -6, @ProcessDate)

        -- Rule 1: multi-branch CASE - penalty scales with how many times
        -- the account has already been dishonoured
        UPDATE D
        SET D.PenaltyAmount = (
                CASE
                    WHEN D.DishonourReason IS NULL THEN NULL
                    WHEN D.RepeatCount IS NULL OR D.RepeatCount = 0 THEN 350.00
                    WHEN D.RepeatCount = 1 THEN 750.00
                    ELSE 1500.00
                END
            )
        FROM PRO.DishonouredCheque D
        WHERE D.DishonourDate = @ProcessDate
          AND D.PenaltyApplied = 'N'

        -- Rule 2: cheques with no reason recorded cannot be penalised and
        -- are held for manual review instead
        UPDATE D
        SET D.HoldForReview = 'Y'
        FROM PRO.DishonouredCheque D
        WHERE D.DishonourDate = @ProcessDate
          AND D.DishonourReason IS NULL

        -- Rule 3: apply the computed penalty to the account balance and
        -- mark the cheque as processed
        UPDATE A
        SET A.OutstandingBalance = ISNULL(A.OutstandingBalance, 0) + D.PenaltyAmount
        FROM PRO.LoanAccountCal A
        INNER JOIN PRO.DishonouredCheque D ON D.AccountId = A.AccountId
        WHERE D.DishonourDate = @ProcessDate
          AND D.PenaltyAmount IS NOT NULL

        UPDATE D
        SET D.PenaltyApplied = 'Y'
        FROM PRO.DishonouredCheque D
        WHERE D.DishonourDate = @ProcessDate
          AND D.PenaltyAmount IS NOT NULL

        IF OBJECT_ID('tempdb..#PenaltyStaging') IS NOT NULL
            DROP TABLE #PenaltyStaging

        CREATE TABLE #PenaltyStaging
        (
            AccountId     VARCHAR(20),
            PenaltyAmount DECIMAL(18,2),
            DishonourCount INT
        )

        -- Rule 4: conditional INSERT - stage only accounts penalised
        -- today, together with their trailing dishonour count
        INSERT INTO #PenaltyStaging (AccountId, PenaltyAmount, DishonourCount)
        SELECT D.AccountId, D.PenaltyAmount,
               (SELECT COUNT(*) FROM PRO.DishonouredCheque D2
                WHERE D2.AccountId = D.AccountId AND D2.DishonourDate >= @LookbackWindowStart)
        FROM PRO.DishonouredCheque D
        WHERE D.DishonourDate = @ProcessDate
          AND D.PenaltyAmount IS NOT NULL

        -- Rule 5: upsert the staged penalty into the ledger - refresh
        -- accounts already tracked, add accounts penalised for the first
        -- time this run
        MERGE PRO.ChequePenaltyLedger AS Target
        USING #PenaltyStaging AS Source
        ON Target.AccountId = Source.AccountId
        WHEN MATCHED THEN
            UPDATE SET
                Target.PenaltyAmount = Target.PenaltyAmount + Source.PenaltyAmount,
                Target.DishonourCount = Source.DishonourCount,
                Target.LastPenaltyDate = @ProcessDate
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (AccountId, PenaltyAmount, DishonourCount, FirstPenaltyDate, LastPenaltyDate)
            VALUES (Source.AccountId, Source.PenaltyAmount, Source.DishonourCount, @ProcessDate, @ProcessDate);

        -- Rule 6: identify accounts crossing the repeat-dishonour
        -- threshold that are not already suspended - NOT EXISTS becomes a
        -- direct NULL/value filter, set-based
        IF OBJECT_ID('tempdb..#NewSuspensions') IS NOT NULL
            DROP TABLE #NewSuspensions

        SELECT P.AccountId
        INTO #NewSuspensions
        FROM #PenaltyStaging P
        INNER JOIN PRO.LoanAccountCal A ON A.AccountId = P.AccountId
        WHERE P.DishonourCount >= 3
          AND (A.ChequeBookSuspended <> 'Y' OR A.ChequeBookSuspended IS NULL)

        -- Rule 7: suspend cheque-book privileges for the newly-identified
        -- accounts and audit each suspension applied
        UPDATE PRO.LoanAccountCal
        SET ChequeBookSuspended = 'Y'
        WHERE AccountId IN (SELECT AccountId FROM #NewSuspensions)

        INSERT INTO PRO.AccountStatusAuditLog (AccountId, TransitionDate, NewStatus, Reason)
        SELECT AccountId, @ProcessDate, 'CHEQUE_BOOK_SUSPENDED', 'REPEAT_DISHONOUR_THRESHOLD'
        FROM #NewSuspensions

        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'Y', ERRORDATE = NULL, ERRORDESCRIPTION = NULL, COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Dishonoured_Cheque_Penalty_Calc'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to
        -- investigate, including a fallback insert if the status row
        -- itself is missing
        IF EXISTS (SELECT 1 FROM PRO.ACLRUNNINGPROCESSSTATUS WHERE RUNNINGPROCESSNAME = 'Dishonoured_Cheque_Penalty_Calc')
        BEGIN
            UPDATE PRO.ACLRUNNINGPROCESSSTATUS
            SET COMPLETED = 'N', ERRORDATE = GETDATE(), ERRORDESCRIPTION = ERROR_MESSAGE(), COUNT = ISNULL(COUNT, 0) + 1
            WHERE RUNNINGPROCESSNAME = 'Dishonoured_Cheque_Penalty_Calc'
        END
        ELSE
        BEGIN
            INSERT INTO PRO.ACLRUNNINGPROCESSSTATUS (RUNNINGPROCESSNAME, COMPLETED, ERRORDATE, ERRORDESCRIPTION, COUNT)
            VALUES ('Dishonoured_Cheque_Penalty_Calc', 'N', GETDATE(), ERROR_MESSAGE(), 1)
        END
    END CATCH
    SET NOCOUNT OFF
END
GO
