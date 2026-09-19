USE [DEMO_MISDB]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO

/*=========================================
 DESCRIPTION : Determines the provisioning percentage by asset class,
               adds an unsecured-exposure loading, calculates the
               provision amount, caps it at the outstanding balance, and
               flags accounts whose provision needs senior review.
 EXEC PRO.Provision_Percentage_Calculation @TimeKey = 25140
=============================================*/
CREATE PROCEDURE PRO.Provision_Percentage_Calculation
    @TimeKey INT
AS
BEGIN
    SET NOCOUNT ON
    BEGIN TRY

        -- Rule 1: base provisioning percentage by asset classification
        UPDATE A
        SET A.ProvisionPct = (
                CASE A.AssetClass
                    WHEN 'STANDARD'    THEN 0.40
                    WHEN 'SUBSTANDARD' THEN 15.00
                    WHEN 'DOUBTFUL'    THEN 25.00
                    WHEN 'LOSS'        THEN 100.00
                    ELSE 0.00
                END
            )
        FROM PRO.AccountCal A

        -- Rule 2: unsecured accounts carry an additional 10-point loading
        UPDATE A
        SET A.ProvisionPct = A.ProvisionPct + 10.00
        FROM PRO.AccountCal A
        WHERE A.SecuredFlag = 'N'
          AND A.AssetClass <> 'STANDARD'

        -- Rule 3 (formula, same shape as the real AddlProvision calculation in
        -- PRO.UpdateNetBalance_AccountWise / InsertDataforAssetClassficationRBL_MOC:
        -- balance * percent / 100): calculate the provision amount
        UPDATE A
        SET A.ProvisionAmount = (A.OutstandingBalance * A.ProvisionPct) / 100
        FROM PRO.AccountCal A
        WHERE A.OutstandingBalance > 0

        -- Rule 4: the provision amount can never exceed the outstanding balance
        UPDATE A
        SET A.ProvisionAmount = A.OutstandingBalance
        FROM PRO.AccountCal A
        WHERE A.ProvisionAmount > A.OutstandingBalance

        -- Rule 5: flag large provisions (over 1,000,000) for senior review
        UPDATE A
        SET A.SeniorReviewFlag = 'Y'
        FROM PRO.AccountCal A
        WHERE A.ProvisionAmount > 1000000

        UPDATE PRO.RunStatus
        SET COMPLETED = 'Y', ErrorDate = NULL, ErrorDescription = NULL, RunCount = ISNULL(RunCount, 0) + 1
        WHERE ProcessName = 'Provision_Percentage_Calculation'

    END TRY
    BEGIN CATCH
        -- Exception handling: record the failure for operations to investigate
        UPDATE PRO.RunStatus
        SET COMPLETED = 'N', ErrorDate = GETDATE(), ErrorDescription = ERROR_MESSAGE(), RunCount = ISNULL(RunCount, 0) + 1
        WHERE ProcessName = 'Provision_Percentage_Calculation'
    END CATCH
END
GO
