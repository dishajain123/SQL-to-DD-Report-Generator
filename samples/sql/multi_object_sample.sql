-- Synthetic sample: two procedures in a single file, used to test
-- Object Split & Classification (architecture step 4) against the
-- "one file, multiple objects" case explicitly.

CREATE OR REPLACE PROCEDURE PRO.Reset_Flags(p_TIMEKEY IN NUMBER) AS
BEGIN
    UPDATE PRO.AccountCal_Stg
    SET FlgDeg = 'N'
    WHERE FlgDeg IS NULL;
END PRO.Reset_Flags;
/

CREATE OR REPLACE FUNCTION PRO.Get_Risk_Band(p_dpd IN NUMBER) RETURN VARCHAR2 AS
    v_band VARCHAR2(20);
BEGIN
    IF p_dpd > 90 THEN
        v_band := 'NPA';
    ELSIF p_dpd > 30 THEN
        v_band := 'SMA';
    ELSE
        v_band := 'STANDARD';
    END IF;
    RETURN v_band;
END PRO.Get_Risk_Band;
/
