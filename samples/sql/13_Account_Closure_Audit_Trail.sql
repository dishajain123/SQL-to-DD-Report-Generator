USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Transitions eligible accounts through the closure workflow
               (PENDING_CLOSURE -> CLOSED), rejects closure for accounts
               that still carry a balance or an unresolved dispute,
               stages and upserts every decision into the closure
               register, walks rejected accounts to notify the owning
               branch, and audits every transition applied.
 EXEC PRO.Account_Closure_Audit_Trail @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.Account_Closure_Audit_Trail
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        DECLARE @ProcessDate DATE = (SELECT [Date] FROM SysDayMatrix WHERE TimeKey = @TimeKey)
        DECLARE @DisputeGraceCutoff DATE = DATEADD(DAY, -30, @ProcessDate)

        -- Rule 1: nested IF/ELSE - accounts pending closure with a zero
        -- or null balance are checked for disputes before closing
        UPDATE A
        SET A.AccountStatus =
            CASE
                WHEN ISNULL(A.OutstandingBalance, 0) = 0 THEN
                    CASE
                        WHEN A.DisputeFlag = 'N' OR A.DisputeFlag IS NULL THEN 'CLOSED'
                        WHEN A.DisputeRaisedDate IS NOT NULL AND A.DisputeRaisedDate <= @DisputeGraceCutoff THEN 'CLOSED'
                        ELSE 'PENDING_CLOSURE'
                    END
                ELSE 'PENDING_CLOSURE'
            END,
            A.ClosureDate = CASE
                WHEN ISNULL(A.OutstandingBalance, 0) = 0
                     AND (A.DisputeFlag = 'N' OR A.DisputeFlag IS NULL
                          OR (A.DisputeRaisedDate IS NOT NULL AND A.DisputeRaisedDate <= @DisputeGraceCutoff))
                THEN @ProcessDate
                ELSE A.ClosureDate
            END
        FROM PRO.LoanAccountCal A
        WHERE A.AccountStatus = 'PENDING_CLOSURE'

        -- Rule 2: accounts pending closure that still carry a balance are
        -- rejected back to ACTIVE with a reason recorded
        UPDATE A
        SET A.AccountStatus = 'ACTIVE',
            A.ClosureRejectReason = 'OUTSTANDING_BALANCE'
        FROM PRO.LoanAccountCal A
        WHERE A.AccountStatus = 'PENDING_CLOSURE'
          AND ISNULL(A.OutstandingBalance, 0) > 0

        -- Rule 3: accounts pending closure with an unresolved dispute
        -- still within the grace window are rejected with a different
        -- reason
        UPDATE A
        SET A.AccountStatus = 'ACTIVE',
            A.ClosureRejectReason = 'UNRESOLVED_DISPUTE'
        FROM PRO.LoanAccountCal A
        WHERE A.AccountStatus = 'PENDING_CLOSURE'
          AND ISNULL(A.OutstandingBalance, 0) = 0
          AND A.DisputeFlag = 'Y'
          AND (A.DisputeRaisedDate IS NULL OR A.DisputeRaisedDate > @DisputeGraceCutoff)

        IF OBJECT_ID('tempdb..#ClosureDecisions') IS NOT NULL
            DROP TABLE #ClosureDecisions

        CREATE TABLE #ClosureDecisions
        (
            AccountId    VARCHAR(20),
            NewStatus    VARCHAR(20),
            Reason       VARCHAR(30)
        )

        -- Rule 4: conditional INSERT - stage only accounts that actually
        -- transitioned status this run
        INSERT INTO #ClosureDecisions (AccountId, NewStatus, Reason)
        SELECT A.AccountId, A.AccountStatus,
               ISNULL(A.ClosureRejectReason, 'CLOSED_ZERO_BALANCE')
        FROM PRO.LoanAccountCal A
        WHERE A.ClosureDate = @ProcessDate
           OR A.ClosureRejectReason IN ('OUTSTANDING_BALANCE', 'UNRESOLVED_DISPUTE')

        -- Rule 5: upsert every decision into the closure register -
        -- refresh accounts already tracked, add accounts decided for the
        -- first time this run
        MERGE PRO.ClosureRegister AS Target
        USING #ClosureDecisions AS Source
        ON Target.AccountId = Source.AccountId
        WHEN MATCHED THEN
            UPDATE SET
                Target.NewStatus = Source.NewStatus,
                Target.Reason = Source.Reason,
                Target.LastDecisionDate = @ProcessDate
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (AccountId, NewStatus, Reason, FirstDecisionDate, LastDecisionDate)
            VALUES (Source.AccountId, Source.NewStatus, Source.Reason, @ProcessDate, @ProcessDate);

        -- Rule 6: notify the owning branch for every rejected account,
        -- read back from the staging table populated above
        INSERT INTO PRO.CollectionsQueue (AccountId, EscalationDate, Reason)
        SELECT AccountId, @ProcessDate, 'CLOSURE_REJECTED_' + Reason
        FROM #ClosureDecisions
        WHERE Reason <> 'CLOSED_ZERO_BALANCE'

        -- Rule 7: every account that transitioned status this run gets an
        -- audit row, cross-referencing the register above
        INSERT INTO PRO.AccountStatusAuditLog (AccountId, TransitionDate, NewStatus, Reason)
        SELECT AccountId, @ProcessDate, NewStatus, Reason
        FROM #ClosureDecisions

        -- Rule 8: clear the reject reason for accounts that are not
        -- currently in a rejected state, so stale reasons do not linger
        UPDATE A
        SET A.ClosureRejectReason = NULL
        FROM PRO.LoanAccountCal A
        WHERE A.AccountStatus NOT IN ('ACTIVE')
           OR A.ClosureRejectReason IS NULL

        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'Y', ERRORDATE = NULL, ERRORDESCRIPTION = NULL, COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Account_Closure_Audit_Trail'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'N', ERRORDATE = GETDATE(), ERRORDESCRIPTION = ERROR_MESSAGE(), COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Account_Closure_Audit_Trail'
    END CATCH
    SET NOCOUNT OFF
END
GO
