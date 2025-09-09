CREATE or replace VIEW processed_demand AS
SELECT * FROM (
    SELECT 
        DATETIME AT TIME ZONE 'Australia/Sydney' AS datetime_au,
        REGIONID,
        AVG(TOTALDEMAND) OVER (
            PARTITION BY REGIONID 
            ORDER BY DATETIME 
            ROWS BETWEEN CURRENT ROW AND 47 FOLLOWING
        ) AS avg_30_min_demand_next_24h,
        MIN(TOTALDEMAND) OVER (
            PARTITION BY REGIONID 
            ORDER BY DATETIME 
            ROWS BETWEEN CURRENT ROW AND 47 FOLLOWING
        ) AS min_30_min_demand_next_24h,
        MAX(TOTALDEMAND) OVER (
            PARTITION BY REGIONID 
            ORDER BY DATETIME 
            ROWS BETWEEN CURRENT ROW AND 47 FOLLOWING
        ) AS max_30_min_demand_next_24h,
        SUM(TOTALDEMAND) OVER (
            PARTITION BY REGIONID 
            ORDER BY DATETIME 
            ROWS BETWEEN CURRENT ROW AND 47 FOLLOWING
        ) AS sum_30_min_demand_next_24h
    FROM totaldemand_nsw
    WHERE datetime >= '2015-12-31 13:00:00'
      AND datetime <= '2019-12-31 13:00:00'
) sub
WHERE EXTRACT(HOUR FROM datetime_au) = 0
  AND EXTRACT(MINUTE FROM datetime_au) = 0;

CREATE or replace VIEW processed_temperature AS
SELECT DISTINCT ON (location, datetime_au) * FROM (
    SELECT 
        DATETIME AT TIME ZONE 'Australia/Sydney' AS datetime_au,
        LOCATION,
        -- Aggregate statistics for next 24 hours
        AVG(TEMPERATURE) OVER (
            PARTITION BY LOCATION 
            ORDER BY DATETIME 
            RANGE BETWEEN CURRENT ROW AND INTERVAL '24 HOURS' FOLLOWING
        ) AS avg_temp_next_24h,
        MIN(TEMPERATURE) OVER (
            PARTITION BY LOCATION 
            ORDER BY DATETIME 
            RANGE BETWEEN CURRENT ROW AND INTERVAL '24 HOURS' FOLLOWING
        ) AS min_temp_next_24h,
        MAX(TEMPERATURE) OVER (
            PARTITION BY LOCATION 
            ORDER BY DATETIME 
            RANGE BETWEEN CURRENT ROW AND INTERVAL '24 HOURS' FOLLOWING
        ) AS max_temp_next_24h,
        -- HDD and CDD calculations for next 24 hours average
        GREATEST(17 - AVG(TEMPERATURE) OVER (
            PARTITION BY LOCATION 
            ORDER BY DATETIME 
            RANGE BETWEEN CURRENT ROW AND INTERVAL '24 HOURS' FOLLOWING
        ), 0) AS hd_next_24h,
        GREATEST(AVG(TEMPERATURE) OVER (
            PARTITION BY LOCATION 
            ORDER BY DATETIME 
            RANGE BETWEEN CURRENT ROW AND INTERVAL '24 HOURS' FOLLOWING
        ) - 19.5, 0) AS cd_next_24h
    FROM temperature_nsw
    WHERE datetime >= '2015-12-31 13:00:00'
      AND datetime <= '2019-12-31 13:00:00'
) sub
WHERE EXTRACT(HOUR FROM datetime_au) = 0
  AND EXTRACT(MINUTE FROM datetime_au) = 0
ORDER BY location, datetime_au, avg_temp_next_24h DESC;

create or replace view processed as
select *,
    CASE WHEN EXTRACT(MONTH FROM datetime_au) IN (12,1,2) THEN 1 ELSE 0 END AS is_summer,
    CASE WHEN EXTRACT(MONTH FROM datetime_au) IN (3,4,5) THEN 1 ELSE 0 END AS is_autumn,
    CASE WHEN EXTRACT(MONTH FROM datetime_au) IN (6,7,8) THEN 1 ELSE 0 END AS is_winter,
    CASE WHEN EXTRACT(MONTH FROM datetime_au) IN (9,10,11) THEN 1 ELSE 0 END AS is_spring,
    -- One-hot encoding for day of week
    CASE WHEN EXTRACT(DOW FROM datetime_au) = 0 THEN 1 ELSE 0 END AS is_sunday,
    CASE WHEN EXTRACT(DOW FROM datetime_au) = 1 THEN 1 ELSE 0 END AS is_monday,
    CASE WHEN EXTRACT(DOW FROM datetime_au) = 2 THEN 1 ELSE 0 END AS is_tuesday,
    CASE WHEN EXTRACT(DOW FROM datetime_au) = 3 THEN 1 ELSE 0 END AS is_wednesday,
    CASE WHEN EXTRACT(DOW FROM datetime_au) = 4 THEN 1 ELSE 0 END AS is_thursday,
    CASE WHEN EXTRACT(DOW FROM datetime_au) = 5 THEN 1 ELSE 0 END AS is_friday,
    CASE WHEN EXTRACT(DOW FROM datetime_au) = 6 THEN 1 ELSE 0 END AS is_saturday,
    -- Is weekend
    CASE WHEN EXTRACT(DOW FROM datetime_au) IN (0,6) THEN 1 ELSE 0 END AS is_weekend,
    -- One-hot encoding for month
    CASE WHEN EXTRACT(MONTH FROM datetime_au) = 1 THEN 1 ELSE 0 END AS is_jan,
    CASE WHEN EXTRACT(MONTH FROM datetime_au) = 2 THEN 1 ELSE 0 END AS is_feb,
    CASE WHEN EXTRACT(MONTH FROM datetime_au) = 3 THEN 1 ELSE 0 END AS is_mar,
    CASE WHEN EXTRACT(MONTH FROM datetime_au) = 4 THEN 1 ELSE 0 END AS is_apr,
    CASE WHEN EXTRACT(MONTH FROM datetime_au) = 5 THEN 1 ELSE 0 END AS is_may,
    CASE WHEN EXTRACT(MONTH FROM datetime_au) = 6 THEN 1 ELSE 0 END AS is_jun,
    CASE WHEN EXTRACT(MONTH FROM datetime_au) = 7 THEN 1 ELSE 0 END AS is_jul,
    CASE WHEN EXTRACT(MONTH FROM datetime_au) = 8 THEN 1 ELSE 0 END AS is_aug,
    CASE WHEN EXTRACT(MONTH FROM datetime_au) = 9 THEN 1 ELSE 0 END AS is_sep,
    CASE WHEN EXTRACT(MONTH FROM datetime_au) = 10 THEN 1 ELSE 0 END AS is_oct,
    CASE WHEN EXTRACT(MONTH FROM datetime_au) = 11 THEN 1 ELSE 0 END AS is_nov,
    CASE WHEN EXTRACT(MONTH FROM datetime_au) = 12 THEN 1 ELSE 0 END AS is_dec
from processed_demand
left join processed_temperature
using (datetime_au);

-- Create a common timezone-converted datetime field
