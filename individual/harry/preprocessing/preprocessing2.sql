-- 1) days: calendar dates + a timestamptz for midnight local time
CREATE OR REPLACE VIEW days AS
SELECT
  d::date AS date_au,
  (d::timestamp AT TIME ZONE 'Australia/Sydney') AS datetime_au_start_tz -- timestamptz for midnight local
FROM generate_series(
  '2016-01-01'::date,
  '2019-12-31'::date,
  '1 day'::interval
) AS gs(d);

-- (optional) 30-min intervals in absolute time (not used by daily aggregation but kept if you need it)
CREATE OR REPLACE VIEW intervals AS
SELECT generate_series(
    '2016-01-01 00:00:00+00'::timestamptz,  -- use UTC here to avoid DST gaps if you want continuous real-time slots
    '2019-12-31 23:30:00+00'::timestamptz,
    '30 mins'::interval
) AS interval_time_utc;

-- 2) clean 30-min demand (remove nulls / average duplicates)
CREATE OR REPLACE VIEW demand AS
SELECT
  datetime,                     -- timestamptz as stored
  AVG(totaldemand) AS demand
FROM totaldemand_nsw
WHERE datetime IS NOT NULL
GROUP BY datetime
ORDER BY datetime;

-- 3) daily aggregates from demand (grouping by local AU date to handle DST correctly)
DROP MATERIALIZED VIEW IF EXISTS processed_demand;
CREATE MATERIALIZED VIEW processed_demand AS
SELECT
  d.date_au,
  AVG(t.demand) AS avg_30_min_demand,
  MIN(t.demand) AS min_30_min_demand,
  MAX(t.demand) AS max_30_min_demand,
  SUM(t.demand) AS sum_30_min_demand,
  COUNT(t.demand) AS count_30_min_points
FROM days d
LEFT JOIN demand t
  ON (t.datetime AT TIME ZONE 'Australia/Sydney')::date = d.date_au
GROUP BY d.date_au
ORDER BY d.date_au;

-- 4) temperature: aggregate by local AU date
-- assume temperature_nsw.datetime is timestamptz
CREATE OR REPLACE VIEW temp_sydney AS
SELECT datetime, temperature FROM temperature_nsw;

DROP MATERIALIZED VIEW IF EXISTS processed_temperature;
CREATE MATERIALIZED VIEW processed_temperature AS
SELECT
  d.date_au,
  AVG(t.temperature) AS avg_temp,
  MIN(t.temperature) AS min_temp,
  MAX(t.temperature) AS max_temp,
  GREATEST(17 - AVG(t.temperature), 0) AS hd_next_24h,   -- heating degree proxy
  GREATEST(AVG(t.temperature) - 19.5, 0) AS cd_next_24h   -- cooling degree proxy
FROM days d
LEFT JOIN temp_sydney t
  ON (t.datetime AT TIME ZONE 'Australia/Sydney')::date = d.date_au
GROUP BY d.date_au
ORDER BY d.date_au;

-- 5) precipitation & sunlight: convert to local date and expose date_au for joining
CREATE OR REPLACE VIEW processed_precipitation AS
SELECT
  (datetime AT TIME ZONE 'Australia/Sydney')::date AS date_au,
  precipitation
FROM precipitation_nsw;

CREATE OR REPLACE VIEW processed_sunlight AS
SELECT
  (datetime AT TIME ZONE 'Australia/Sydney')::date AS date_au,
  sunlight
FROM sunlight_nsw;

-- 6) final processed daily view: combine everything and add features
CREATE OR REPLACE VIEW processed2 AS
SELECT
  d.date_au AS datetime_au,    -- keep the column name you wanted; this is a date (local AU)
  -- seasonal flags
  CASE WHEN EXTRACT(MONTH FROM d.date_au) IN (12,1,2) THEN 1 ELSE 0 END AS is_summer,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) IN (3,4,5) THEN 1 ELSE 0 END AS is_autumn,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) IN (6,7,8) THEN 1 ELSE 0 END AS is_winter,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) IN (9,10,11) THEN 1 ELSE 0 END AS is_spring,
  -- day-of-week one-hot
  CASE WHEN EXTRACT(DOW FROM d.date_au) = 0 THEN 1 ELSE 0 END AS is_sunday,
  CASE WHEN EXTRACT(DOW FROM d.date_au) = 1 THEN 1 ELSE 0 END AS is_monday,
  CASE WHEN EXTRACT(DOW FROM d.date_au) = 2 THEN 1 ELSE 0 END AS is_tuesday,
  CASE WHEN EXTRACT(DOW FROM d.date_au) = 3 THEN 1 ELSE 0 END AS is_wednesday,
  CASE WHEN EXTRACT(DOW FROM d.date_au) = 4 THEN 1 ELSE 0 END AS is_thursday,
  CASE WHEN EXTRACT(DOW FROM d.date_au) = 5 THEN 1 ELSE 0 END AS is_friday,
  CASE WHEN EXTRACT(DOW FROM d.date_au) = 6 THEN 1 ELSE 0 END AS is_saturday,
  CASE WHEN EXTRACT(DOW FROM d.date_au) IN (0,6) THEN 1 ELSE 0 END AS is_weekend,
  CASE WHEN EXTRACT(DOW FROM d.date_au) IN (1,2,3,4,5) THEN 1 ELSE 0 END AS is_weekday,
  -- month one-hot
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 1 THEN 1 ELSE 0 END AS is_jan,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 2 THEN 1 ELSE 0 END AS is_feb,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 3 THEN 1 ELSE 0 END AS is_mar,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 4 THEN 1 ELSE 0 END AS is_apr,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 5 THEN 1 ELSE 0 END AS is_may,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 6 THEN 1 ELSE 0 END AS is_jun,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 7 THEN 1 ELSE 0 END AS is_jul,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 8 THEN 1 ELSE 0 END AS is_aug,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 9 THEN 1 ELSE 0 END AS is_sep,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 10 THEN 1 ELSE 0 END AS is_oct,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 11 THEN 1 ELSE 0 END AS is_nov,
  CASE WHEN EXTRACT(MONTH FROM d.date_au) = 12 THEN 1 ELSE 0 END AS is_dec,
  -- metrics joined from the processed materialized views / views
  pd.avg_30_min_demand,
  pd.min_30_min_demand,
  pd.max_30_min_demand,
  pd.sum_30_min_demand,
  pd.count_30_min_points,

  pt.avg_temp,
  pt.min_temp,
  pt.max_temp,
  pt.hd_next_24h,
  pt.cd_next_24h,

  pr.precipitation,
  ps.sunlight

FROM days d
LEFT JOIN processed_demand pd USING (date_au)
LEFT JOIN processed_temperature pt USING (date_au)
LEFT JOIN processed_precipitation pr USING (date_au)
LEFT JOIN processed_sunlight ps USING (date_au)
ORDER BY d.date_au;

-- Missing Temperatures
SELECT d.date_au
FROM days d
LEFT JOIN temperature_nsw t
  ON (t.datetime AT TIME ZONE 'Australia/Sydney')::date = d.date_au
GROUP BY d.date_au
HAVING COUNT(t.*) = 0
ORDER BY d.date_au;


COPY (select * from processed)
TO '/import/processed.csv'
WITH (
    FORMAT CSV,
    HEADER true,
    DELIMITER ','
);




