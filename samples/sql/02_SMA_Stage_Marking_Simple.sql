USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Marks each account's SMA (Special Mention Account) stage,
               flags SMA status, attributes a reason by facility type,
               calculates the date the current stage began, and clears
               SMA status for accounts no longer overdue.
 EXEC PRO.SMA_Stage_Marking_Simple @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.SMA_Stage_Marking_Simple
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        DECLARE @ProcessDate DATE = (SELECT [Date] FROM SysDayMatrix WHERE TimeKey = @TimeKey)

        -- Rule 1: classify the SMA stage by overdue days
        UPDATE A
        SET A.SmaStage = (
                CASE
                    WHEN A.OverdueDays BETWEEN 1 AND 30 THEN 'SMA_0'
                    WHEN A.OverdueDays BETWEEN 31 AND 60 THEN 'SMA_1'
                    WHEN A.OverdueDays BETWEEN 61 AND 90 THEN 'SMA_2'
                    ELSE NULL
                END
            )
        FROM PRO.AccountCal A
        WHERE A.OverdueDays > 0

        -- Rule 2: flag the account as currently in SMA status
        UPDATE A
        SET A.FlagSma = (
                CASE
                    WHEN A.SmaStage IS NOT NULL THEN 'Y'
                    ELSE 'N'
                END
            )
        FROM PRO.AccountCal A

        -- Rule 3: attribute the overdue reason by facility type, for SMA accounts
        UPDATE A
        SET A.SmaReason = (
                CASE
                    WHEN A.FacilityType IN ('CC', 'OD') THEN 'CASH CREDIT / OVERDRAFT OVERDUE'
                    WHEN A.FacilityType IN ('TL', 'DL') THEN 'TERM LOAN OVERDUE'
                    ELSE 'OTHER FACILITY OVERDUE'
                END
            )
        FROM PRO.AccountCal A
        WHERE A.FlagSma = 'Y'

        -- Rule 4 (formula, taken directly from PRO.SMA_MARKING's real
        -- `DATEADD(DAY, -dpd.DPD_MAX+1, @ProcessDate)` calculation):
        -- back-calculate the date the current SMA stage began
        UPDATE A
        SET A.SmaStageDate = DATEADD(DAY, -A.OverdueDays + 1, @ProcessDate)
        FROM PRO.AccountCal A
        WHERE A.FlagSma = 'Y'

        -- Rule 5: clear SMA status for accounts that are no longer overdue
        UPDATE A
        SET A.SmaStage = NULL,
            A.FlagSma = 'N',
            A.SmaReason = NULL
        FROM PRO.AccountCal A
        WHERE A.OverdueDays = 0

        UPDATE PRO.RunStatus
        SET COMPLETED = 'Y', ErrorDate = NULL, ErrorDescription = NULL, RunCount = ISNULL(RunCount, 0) + 1
        WHERE ProcessName = 'SMA_Stage_Marking_Simple'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.RunStatus
        SET COMPLETED = 'N', ErrorDate = GETDATE(), ErrorDescription = ERROR_MESSAGE(), RunCount = ISNULL(RunCount, 0) + 1
        WHERE ProcessName = 'SMA_Stage_Marking_Simple'
    END CATCH
END
GO
