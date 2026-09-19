USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : End-to-end NPA classification for active accounts -
               classifies asset class by overdue days, flags NPA status,
               calculates additional provisioning for NPA accounts,
               stamps the classification date, and flags accounts for
               restructuring review.
 EXEC PRO.NPA_Classification_Simple @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.NPA_Classification_Simple
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        DECLARE @ProcessDate DATE = (SELECT [Date] FROM SysDayMatrix WHERE TimeKey = @TimeKey)

        -- Rule 1: classify the account's asset class by days-past-due
        UPDATE A
        SET A.AssetClass = (
                CASE
                    WHEN A.DaysPastDue <= 90 THEN 'STANDARD'
                    WHEN A.DaysPastDue BETWEEN 91 AND 180 THEN 'SUBSTANDARD'
                    WHEN A.DaysPastDue BETWEEN 181 AND 365 THEN 'DOUBTFUL'
                    ELSE 'LOSS'
                END
            )
        FROM PRO.AccountCal A
        WHERE A.AccountStatus = 'ACTIVE'

        -- Rule 2: flag the account as NPA whenever it isn't Standard
        UPDATE A
        SET A.NpaFlag = (
                CASE
                    WHEN A.AssetClass = 'STANDARD' THEN 'N'
                    ELSE 'Y'
                END
            )
        FROM PRO.AccountCal A
        WHERE A.AccountStatus = 'ACTIVE'

        -- Rule 3 (formula, sourced from PRO.UpdateNetBalance_AccountWise /
        -- InsertDataforAssetClassficationAUSFB_MOC / InsertDataforAssetClassficationRBL_MOC,
        -- where this exact shape appears in your production corpus):
        -- additional provision = balance * additional-provision-percent / 100
        UPDATE A
        SET A.AddlProvision = (A.OutstandingBalance * A.AddlProvisionPer) / 100
        FROM PRO.AccountCal A
        WHERE A.NpaFlag = 'Y'
          AND ISNULL(A.AddlProvisionPer, 0) <> 0

        -- Rule 4: stamp the classification date on accounts newly marked NPA this run
        UPDATE A
        SET A.ClassificationDate = @ProcessDate
        FROM PRO.AccountCal A
        WHERE A.NpaFlag = 'Y'
          AND A.ClassificationDate IS NULL

        -- Rule 5: flag accounts approaching NPA (61-90 days overdue) for restructuring review
        UPDATE A
        SET A.RestructureReviewFlag = 'Y'
        FROM PRO.AccountCal A
        WHERE A.DaysPastDue BETWEEN 61 AND 90

        UPDATE PRO.RunStatus
        SET COMPLETED = 'Y', ErrorDate = NULL, ErrorDescription = NULL, RunCount = ISNULL(RunCount, 0) + 1
        WHERE ProcessName = 'NPA_Classification_Simple'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.RunStatus
        SET COMPLETED = 'N', ErrorDate = GETDATE(), ErrorDescription = ERROR_MESSAGE(), RunCount = ISNULL(RunCount, 0) + 1
        WHERE ProcessName = 'NPA_Classification_Simple'
    END CATCH
END
GO
