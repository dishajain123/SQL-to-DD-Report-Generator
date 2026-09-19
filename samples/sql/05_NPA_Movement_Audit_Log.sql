USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Records an account-level and customer-level audit trail
               whenever an account's asset classification changes,
               refreshes the "last known classification" snapshot,
               calculates the running movement count, and flags
               customers with multiple accounts moving in one run.
 EXEC PRO.NPA_Movement_Audit_Log @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.NPA_Movement_Audit_Log
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        -- Rule 1 (condition modeled on PRO.SMA_MARKING's real change-detection
        -- logic: `WHERE B.SMA_CLASS IS NOT NULL AND ISNULL(A.SMA_CLASS,'')<>ISNULL(B.SMA_CLASS,'')`):
        -- log an account-level movement row only when classification changed
        INSERT INTO PRO.AssetClassMovementHistory (TimeKey, AccountId, PreviousClass, CurrentClass, MovementDate)
        SELECT @TimeKey, B.AccountId, A.AssetClass, B.AssetClass, GETDATE()
        FROM PRO.PreviousAssetClass A
        RIGHT OUTER JOIN PRO.AccountCal B ON A.AccountId = B.AccountId
        WHERE ISNULL(A.AssetClass, '') <> ISNULL(B.AssetClass, '')

        -- Rule 2: also log a customer-level movement row, using the worst account per customer
        INSERT INTO PRO.CustomerClassMovementHistory (TimeKey, CustomerId, WorstClass, MovementDate)
        SELECT @TimeKey, B.CustomerId, MIN(B.AssetClass), GETDATE()
        FROM PRO.AccountCal B
        WHERE B.AssetClass <> 'STANDARD'
        GROUP BY B.CustomerId

        -- Rule 3: refresh the snapshot used to detect next run's changes
        TRUNCATE TABLE PRO.PreviousAssetClass

        INSERT INTO PRO.PreviousAssetClass (AccountId, AssetClass)
        SELECT AccountId, AssetClass
        FROM PRO.AccountCal

        -- Rule 4 (formula): increment the movement counter used for the daily operations dashboard
        UPDATE PRO.RunStatistics
        SET MovementCount = MovementCount + (SELECT COUNT(*) FROM PRO.AssetClassMovementHistory WHERE TimeKey = @TimeKey)
        WHERE StatisticName = 'DAILY_ASSET_MOVEMENTS'

        -- Rule 5: flag customers with more than one account moving this run for manual review
        UPDATE C
        SET C.MultiAccountMovementFlag = 'Y'
        FROM PRO.CustomerCal C
        WHERE C.CustomerId IN (
            SELECT CustomerId
            FROM PRO.CustomerClassMovementHistory
            WHERE TimeKey = @TimeKey
            GROUP BY CustomerId
            HAVING COUNT(*) > 1
        )

        UPDATE PRO.RunStatus
        SET COMPLETED = 'Y', ErrorDate = NULL, ErrorDescription = NULL, RunCount = ISNULL(RunCount, 0) + 1
        WHERE ProcessName = 'NPA_Movement_Audit_Log'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.RunStatus
        SET COMPLETED = 'N', ErrorDate = GETDATE(), ErrorDescription = ERROR_MESSAGE(), RunCount = ISNULL(RunCount, 0) + 1
        WHERE ProcessName = 'NPA_Movement_Audit_Log'
    END CATCH
END
GO
