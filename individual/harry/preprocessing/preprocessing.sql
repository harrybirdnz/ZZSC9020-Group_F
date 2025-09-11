-- create days view
create or replace view days as
select generate_series(
    '2016-01-01 00:00:00+11'::timestamptz,
    '2019-12-31 00:00:00+11'::timestamptz,
    interval '1 day'
) AS datetime_au;

-- create 30 min intervals view
create view intervals as
select generate_series(
    '2016-01-01 00:00:00+11'::timestamptz,
    '2019-12-31 00:00:00+11'::timestamptz,
    interval '30 mins'
) AS interval_time;

-- check for missing demand values
SET TIME ZONE 'Australia/Sydney';
SELECT i.interval_time_utc, t.totaldemand
FROM intervals i
LEFT JOIN totaldemand_nsw t
  ON t.datetime = i.interval_time_utc
WHERE t.totaldemand IS NULL;
-- 16 missing values
-- all at daylight savings time - I assume
-- inspecting the data on these days shows no actual data gaps - they're ghost gaps caused by moving clocks

--  remove duplicates and nulls
create view demand as (
    select datetime, avg(totaldemand) as demand 
    from totaldemand_nsw
    where datetime is not null
    GROUP BY datetime
    order by datetime
)

DROP MATERIALIZED VIEW processed_demand;
CREATE or replace materialized VIEW processed_demand AS 
SELECT 
    d.datetime_au::date AS datetime_au,
    AVG(t.demand) AS avg_30_min_demand_next_24h,
    MIN(t.demand) AS min_30_min_demand_next_24h,
    MAX(t.demand) AS max_30_min_demand_next_24h,
    SUM(t.demand) AS sum_30_min_demand_next_24h
FROM days d
LEFT JOIN (
    SELECT datetime, AVG(totaldemand) AS demand
    FROM totaldemand_nsw
    WHERE datetime IS NOT NULL
    GROUP BY datetime
    ORDER BY datetime
  ) t
  ON t.datetime >= d.datetime_au
 AND t.datetime < d.datetime_au + interval '1 day'
GROUP BY d.datetime_au
ORDER BY d.datetime_au;

create or replace view temp_sydney AS (
    SELECT
        *, 
        datetime AS datetime_au
    FROM temperature_nsw
)

DROP MATERIALIZED VIEW processed_temperature;
CREATE MATERIALIZED VIEW processed_temperature AS (
SELECT 
    d.datetime_au::date AS datetime_au,
    AVG(t.temperature) AS avg_temp_next_24h,
    MIN(t.temperature) AS min_temp_next_24h,
    MAX(t.temperature) AS max_temp_next_24h,
    GREATEST(17 - AVG(t.temperature), 0) AS hd_next_24h,
    GREATEST(AVG(t.temperature) - 19.5, 0) AS cd_next_24h
FROM days d
LEFT JOIN temp_sydney t
  ON t.datetime_au >= d.datetime_au
 AND t.datetime_au < d.datetime_au + interval '1 day'
GROUP BY d.datetime_au
ORDER BY d.datetime_au);


create or replace view processed_precipitation as
select DATETIME AT TIME ZONE 'Australia/Sydney' AS datetime_au,
precipitation
from precipitation_nsw;

create or replace view processed_sunlight as
select DATETIME AT TIME ZONE 'Australia/Sydney' AS datetime_au,
sunlight
from sunlight_nsw;


create view processed as
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
    CASE WHEN EXTRACT(DOW FROM datetime_au) IN (1,2,3,4,5) THEN 1 ELSE 0 END AS is_weekday,
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
from days
left join processed_demand using (datetime_au)
left join processed_temperature using (datetime_au)
left join processed_precipitation using (datetime_au)
left join processed_sunlight using (datetime_au);




select * from processed
where avg_temp_next_24h is null;


select * from temperature_nsw
where datetime >= '2016-07-15 00:00:00+11'
  and datetime <= '2016-07-24 00:00:00+11';

select * from totaldemand_nsw
where datetime not = row following datetime - 30 mins;

with demand_intervals as (
    SELECT *, LEAD(datetime) OVER (ORDER BY datetime) <> datetime + INTERVAL '30 minutes' as int_not_30
    FROM totaldemand_nsw
)
select * from demand_intervals where int_not_30;

select count(*), datetime from totaldemand_nsw group by datetime having count(*) = 3 order by count(*) desc ;

select * from totaldemand_nsw_staging
where
datetime >= '2016-04-02 00:00:00+11'
and datetime <= '2016-04-02 24:00:00+11';




select * from totaldemand_nsw where totaldemand is null;

COPY totaldemand_nsw
FROM '/import/totaldemand_nsw.csv'
WITH (
    FORMAT CSV,
    HEADER true,
    DELIMITER ','
);

CREATE TABLE totaldemand_nsw_staging(
  datetime TIMESTAMPtz,
  totaldemand NUMERIC,
  regionid text
);

COPY totaldemand_nsw_staging
FROM '/import/totaldemand_nsw_formatted.csv'
WITH (
    FORMAT CSV,
    HEADER true,
    DELIMITER ','
);

COPY processed
TO '/import/processed.csv'
WITH (
    FORMAT CSV,
    HEADER true,
    DELIMITER ','
);

set timezone to 'Australia/Sydney';