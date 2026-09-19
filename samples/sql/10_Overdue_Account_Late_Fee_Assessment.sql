USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Applies a late-fee calculation and notification flag to
               every overdue account, stages and upserts the fee
               schedule into the notification register, and escalates
               repeat offenders to collections.
 EXEC PRO.Overdue_Account_Late_Fee_Assessment @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.Overdue_Account_Late_Fee_Assessment
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        DECLARE @ProcessDate DATE = (SELECT [Date] FROM SysDayMatrix WHERE TimeKey = @TimeKey)
        DECLARE @MonthStartDate DATE = DATEADD(DAY, -DAY(@ProcessDate) + 1, @ProcessDate)
        DECLARE @GraceWindowEnd DATE = DATEADD(DAY, 6, @MonthStartDate)

        IF OBJECT_ID('tempdb..#FeeSchedule') IS NOT NULL
            DROP TABLE #FeeSchedule

        CREATE TABLE #FeeSchedule
        (
            AccountId    VARCHAR(20),
            LateFee      DECIMAL(18,2),
            NotifyCount  INT,
            EscalateFlag VARCHAR(1)
        )

        -- Rule 0: NULL check - accounts with no last-notify date on record
        -- are treated as never notified before, distinct from accounts
        -- explicitly notified zero times
        UPDATE A
        SET A.NotifyCount = 0
        FROM PRO.LoanAccountCal A
        WHERE A.OverdueDays > 0
          AND A.LastNotifyDate IS NULL
          AND A.NotifyCount IS NOT NULL

        -- Rule 1: conditional INSERT - only accounts actually charged a
        -- fee this cycle are staged; nested IF/ELSE (waived entirely
        -- within the first week of the month for a short grace window,
        -- otherwise the fee escalates with how many times the account
        -- has already been notified this cycle) expressed as a
        -- multi-branch CASE, set-based
        INSERT INTO #FeeSchedule (AccountId, LateFee, NotifyCount, EscalateFlag)
        SELECT AccountId,
               CASE
                   WHEN ISNULL(NotifyCount, 0) = 0 THEN 250.00
                   WHEN ISNULL(NotifyCount, 0) = 1 THEN 500.00
                   ELSE 1000.00
               END,
               ISNULL(NotifyCount, 0) + 1,
               NULL
        FROM PRO.LoanAccountCal
        WHERE OverdueDays > 0
          AND NOT (@ProcessDate <= @GraceWindowEnd AND OverdueDays <= 5)

        -- Rule 2: multi-branch CASE - derive the escalation tier for
        -- every staged account, read back from the staging table above
        UPDATE S
        SET S.EscalateFlag = (
                CASE
                    WHEN S.NotifyCount >= 3 THEN 'Y'
                    WHEN S.NotifyCount = 2 THEN 'PENDING'
                    ELSE 'N'
                END
            )
        FROM #FeeSchedule S

        -- Rule 3: derived-assignment UPDATE, apply the staged fee and
        -- record that a notification is due, read back from staging
        UPDATE A
        SET A.LateFeeAmount = ISNULL(A.LateFeeAmount, 0) + S.LateFee,
            A.NotifyCount = S.NotifyCount,
            A.LastNotifyDate = @ProcessDate
        FROM PRO.LoanAccountCal A
        INNER JOIN #FeeSchedule S ON S.AccountId = A.AccountId

        -- Rule 4: upsert the fee schedule into the notification register
        -- - refresh accounts already tracked, add accounts fined for the
        -- first time this run
        MERGE PRO.NotificationRegister AS Target
        USING #FeeSchedule AS Source
        ON Target.AccountId = Source.AccountId
        WHEN MATCHED THEN
            UPDATE SET
                Target.LateFee = Source.LateFee,
                Target.NotifyCount = Source.NotifyCount,
                Target.LastNotifyDate = @ProcessDate
        WHEN NOT MATCHED BY TARGET THEN
            INSERT (AccountId, LateFee, NotifyCount, FirstNotifyDate, LastNotifyDate)
            VALUES (Source.AccountId, Source.LateFee, Source.NotifyCount, @ProcessDate, @ProcessDate);

        -- Rule 5: status transition - accounts notified three or more
        -- times are escalated to the collections queue
        INSERT INTO PRO.CollectionsQueue (AccountId, EscalationDate, Reason)
        SELECT AccountId, @ProcessDate, 'REPEATED_OVERDUE_NOTIFICATION'
        FROM #FeeSchedule
        WHERE EscalateFlag = 'Y'

        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'Y', ERRORDATE = NULL, ERRORDESCRIPTION = NULL, COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Overdue_Account_Late_Fee_Assessment'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.ACLRUNNINGPROCESSSTATUS
        SET COMPLETED = 'N', ERRORDATE = GETDATE(), ERRORDESCRIPTION = ERROR_MESSAGE(), COUNT = ISNULL(COUNT, 0) + 1
        WHERE RUNNINGPROCESSNAME = 'Overdue_Account_Late_Fee_Assessment'
    END CATCH
    SET NOCOUNT OFF
END
GO
