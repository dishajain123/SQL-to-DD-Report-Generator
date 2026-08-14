-- Sample MySQL procedure, used to test dialect detection and MySQL parsing.
DELIMITER $$

CREATE PROCEDURE Customer_Risk_Flag(IN p_as_of_date DATE)
BEGIN
    UPDATE customer_risk
    SET risk_flag = 0
    WHERE as_of_date = p_as_of_date;

    UPDATE customer_risk cr
    SET cr.risk_flag = 1
    WHERE cr.as_of_date = p_as_of_date
      AND IFNULL(cr.days_past_due, 0) > 90;

    UPDATE customer_risk cr
    SET cr.risk_flag = 2
    WHERE cr.as_of_date = p_as_of_date
      AND IFNULL(cr.days_past_due, 0) BETWEEN 31 AND 90;
END$$

DELIMITER ;
