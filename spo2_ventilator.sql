DROP MATERIALIZED VIEW IF EXISTS spo2_ventsettings CASCADE;
CREATE MATERIALIZED VIEW spo2_ventsettings AS
select
  icustay_id, charttime
  -- case statement determining whether it is an instance of mech vent
  , max(
    case
      when itemid is null or value is null then 0 -- can't have null values
      when itemid = 720 and value != 'Other/Remarks' THEN 1  -- VentTypeRecorded
      when itemid = 223848 and value != 'Other' THEN 1
      when itemid = 223849 then 1 -- ventilator mode
      when itemid = 467 and value = 'Ventilator' THEN 1 -- O2 delivery device == ventilator
      when itemid in
        (
        445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
        , 639, 654, 681, 682, 683, 684,224685,224684,224686 -- tidal volume
        , 218,436,535,444,459,224697,224695,224696,224746,224747 -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
        , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 -- Insp pressure
        , 543 -- PlateauPressure
        , 5865,5866,224707,224709,224705,224706 -- APRV pressure
        , 60,437,505,506,686,220339,224700 -- PEEP
        , 3459 -- high pressure relief
        , 501,502,503,224702 -- PCV
        , 223,667,668,669,670,671,672 -- TCPCV
        , 224701 -- PSVlevel
        )
        THEN 1
      else 0
    end
    ) as MechVent
    , max(
      case
        -- initiation of oxygen therapy indicates the ventilation has ended
        when itemid = 226732 and value in
        (
          'Nasal cannula', -- 153714 observations
          'Face tent', -- 24601 observations
          'Aerosol-cool', -- 24560 observations
          'Trach mask ', -- 16435 observations
          'High flow neb', -- 10785 observations
          'Non-rebreather', -- 5182 observations
          'Venti mask ', -- 1947 observations
          'Medium conc mask ', -- 1888 observations
          'T-piece', -- 1135 observations
          'High flow nasal cannula', -- 925 observations
          'Ultrasonic neb', -- 9 observations
          'Vapomist' -- 3 observations
        ) then 1
        when itemid = 467 and value in
        (
          'Cannula', -- 278252 observations
          'Nasal Cannula', -- 248299 observations
          -- 'None', -- 95498 observations
          'Face Tent', -- 35766 observations
          'Aerosol-Cool', -- 33919 observations
          'Trach Mask', -- 32655 observations
          'Hi Flow Neb', -- 14070 observations
          'Non-Rebreather', -- 10856 observations
          'Venti Mask', -- 4279 observations
          'Medium Conc Mask', -- 2114 observations
          'Vapotherm', -- 1655 observations
          'T-Piece', -- 779 observations
          'Hood', -- 670 observations
          'Hut', -- 150 observations
          'TranstrachealCat', -- 78 observations
          'Heated Neb', -- 37 observations
          'Ultrasonic Neb' -- 2 observations
        ) then 1
      else 0
      end
    ) as OxygenTherapy
    , max(
      case when itemid is null or value is null then 0
        -- extubated indicates ventilation event has ended
        when itemid = 640 and value = 'Extubated' then 1
        when itemid = 640 and value = 'Self Extubation' then 1
      else 0
      end
      )
      as Extubated
    , max(
      case when itemid is null or value is null then 0
        when itemid = 640 and value = 'Self Extubation' then 1
      else 0
      end
      )
      as SelfExtubated
from mimiciii.chartevents ce
where ce.value is not null
-- exclude rows marked as error
and ce.error IS DISTINCT FROM 1
and itemid in
(
    -- the below are settings used to indicate ventilation
      720, 223849 -- vent mode
    , 223848 -- vent type
    , 445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
    , 639, 654, 681, 682, 683, 684,224685,224684,224686 -- tidal volume
    , 218,436,535,444,224697,224695,224696,224746,224747 -- High/Low/Peak/Mean ("RespPressure")
    , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 -- Insp pressure
    , 543 -- PlateauPressure
    , 5865,5866,224707,224709,224705,224706 -- APRV pressure
    , 60,437,505,506,686,220339,224700 -- PEEP
    , 3459 -- high pressure relief
    , 501,502,503,224702 -- PCV
    , 223,667,668,669,670,671,672 -- TCPCV
    , 224701 -- PSVlevel

    -- the below are settings used to indicate extubation
    , 640 -- extubated

    -- the below indicate oxygen/NIV, i.e. the end of a mechanical vent event
    , 468 -- O2 Delivery Device#2
    , 469 -- O2 Delivery Mode
    , 470 -- O2 Flow (lpm)
    , 471 -- O2 Flow (lpm) #2
    , 227287 -- O2 Flow (additional cannula)
    , 226732 -- O2 Delivery Device(s)
    , 223834 -- O2 Flow

    -- used in both oxygen + vent calculation
    , 467 -- O2 Delivery Device
)
group by icustay_id, charttime
UNION
-- add in the extubation flags from procedureevents_mv
-- note that we only need the start time for the extubation
-- (extubation is always charted as ending 1 minute after it started)
select
  icustay_id, starttime as charttime
  , 0 as MechVent
  , 0 as OxygenTherapy
  , 1 as Extubated
  , case when itemid = 225468 then 1 else 0 end as SelfExtubated
from mimiciii.procedureevents_mv
where itemid in
(
  227194 -- "Extubation"
, 225468 -- "Unplanned Extubation (patient-initiated)"
, 225477 -- "Unplanned Extubation (non-patient initiated)"
);



DROP MATERIALIZED VIEW IF EXISTS spo2_VENTDURATIONS CASCADE;
create MATERIALIZED VIEW spo2_ventdurations as
with vd0 as
(
  select
    icustay_id
    -- this carries over the previous charttime which had a mechanical ventilation event
    , case
        when MechVent=1 then
          LAG(CHARTTIME, 1) OVER (partition by icustay_id, MechVent order by charttime)
        else null
      end as charttime_lag
    , charttime
    , MechVent
    , OxygenTherapy
    , Extubated
    , SelfExtubated
  from spo2_ventsettings
)
, vd1 as
(
  select
      icustay_id
      , charttime_lag
      , charttime
      , MechVent
      , OxygenTherapy
      , Extubated
      , SelfExtubated

      -- if this is a mechanical ventilation event, we calculate the time since the last event
      , case
          -- if the current observation indicates mechanical ventilation is present
          -- calculate the time since the last vent event
          when MechVent=1 then
            CHARTTIME - charttime_lag
          else null
        end as ventduration

      , LAG(Extubated,1)
      OVER
      (
      partition by icustay_id, case when MechVent=1 or Extubated=1 then 1 else 0 end
      order by charttime
      ) as ExtubatedLag

      -- now we determine if the current mech vent event is a "new", i.e. they've just been intubated
      , case
        -- if there is an extubation flag, we mark any subsequent ventilation as a new ventilation event
          --when Extubated = 1 then 0 -- extubation is *not* a new ventilation event, the *subsequent* row is
          when
            LAG(Extubated,1)
            OVER
            (
            partition by icustay_id, case when MechVent=1 or Extubated=1 then 1 else 0 end
            order by charttime
            )
            = 1 then 1
          -- if patient has initiated oxygen therapy, and is not currently vented, start a newvent
          when MechVent = 0 and OxygenTherapy = 1 then 1
            -- if there is less than 8 hours between vent settings, we do not treat this as a new ventilation event
          when (CHARTTIME - charttime_lag) > interval '8' hour
            then 1
        else 0
        end as newvent
  -- use the staging table with only vent settings from chart events
  FROM vd0 ventsettings
)
, vd2 as
(
  select vd1.*
  -- create a cumulative sum of the instances of new ventilation
  -- this results in a monotonic integer assigned to each instance of ventilation
  , case when MechVent=1 or Extubated = 1 then
      SUM( newvent )
      OVER ( partition by icustay_id order by charttime )
    else null end
    as ventnum
  --- now we convert CHARTTIME of ventilator settings into durations
  from vd1
)
-- create the durations for each mechanical ventilation instance
select icustay_id
  -- regenerate ventnum so it's sequential
  , ROW_NUMBER() over (partition by icustay_id order by ventnum) as ventnum
  , min(charttime) as starttime
  , max(charttime) as endtime
  , extract(epoch from max(charttime)-min(charttime))/60/60 AS duration_hours
from vd2
group by icustay_id, ventnum
having min(charttime) != max(charttime)
-- patient had to be mechanically ventilated at least once
-- i.e. max(mechvent) should be 1
-- this excludes a frequent situation of NIV/oxygen before intub
-- in these cases, ventnum=0 and max(mechvent)=0, so they are ignored
and max(mechvent) = 1
order by icustay_id, ventnum;


-- select mechanically ventilator icustay_id with only one event(the first one)
DROP MATERIALIZED VIEW IF EXISTS spo2_ventdurations_unievent CASCADE;
create MATERIALIZED VIEW spo2_ventdurations_unievent as
select * from spo2_ventdurations where ventnum=1 and icustay_id is not null;


-- creat a view about extract spo2 information in chartevents
DROP MATERIALIZED VIEW IF EXISTS spo2_chartevents CASCADE;
create MATERIALIZED VIEW spo2_chartevents as
select * from mimiciii.chartevents where itemid=646 or itemid=220277;


--chartevents only have first 24 hours spo2 value
DROP MATERIALIZED VIEW IF EXISTS spo2_24hr CASCADE;
create MATERIALIZED VIEW spo2_24hr as
select spo2_chartevents.subject_id,spo2_chartevents.icustay_id,itemid,value,valueuom,charttime, spo2_ventdurations_unievent.starttime,endtime,duration_hours,ventnum
from spo2_chartevents inner join spo2_ventdurations_unievent on spo2_chartevents.icustay_id = spo2_ventdurations_unievent.icustay_id
where case
    when spo2_ventdurations_unievent.endtime>=(spo2_ventdurations_unievent.starttime+interval '1 day')
         then spo2_chartevents.charttime between spo2_ventdurations_unievent.starttime and (spo2_ventdurations_unievent.starttime+interval '1 day')
    when spo2_ventdurations_unievent.endtime<(spo2_ventdurations_unievent.starttime+interval '1 day')
         then spo2_chartevents.charttime between spo2_ventdurations_unievent.starttime and spo2_ventdurations_unievent.endtime
    end;


-- labevents only have pao2
DROP MATERIALIZED VIEW IF EXISTS pao2_lab CASCADE;
create MATERIALIZED VIEW pao2_lab as
select * from mimiciii.labevents
where itemid in (50821, 490);  -- pao2



DROP MATERIALIZED VIEW IF EXISTS spo2_pao2_24hr CASCADE;
create MATERIALIZED VIEW spo2_pao2_24hr as
select spo2_24hr.subject_id,icustay_id,duration_hours,starttime,endtime,spo2_24hr.itemid,spo2_24hr.charttime,spo2_24hr.value,spo2_24hr.valueuom,
       pao2_lab.charttime as labtime,pao2_lab.itemid as pao2_id,pao2_lab.value as pao2_value,pao2_lab.valueuom as pao2_valueuom
from spo2_24hr inner join pao2_lab on spo2_24hr.subject_id= pao2_lab.subject_id
where case
    when spo2_24hr.endtime>=(spo2_24hr.starttime+interval '1 day')
         then pao2_lab.charttime between spo2_24hr.starttime and (spo2_24hr.starttime+interval '1 day')
    when spo2_24hr.endtime<(spo2_24hr.starttime+interval '1 day')
         then pao2_lab.charttime between spo2_24hr.starttime and spo2_24hr.endtime
    end;


DROP MATERIALIZED VIEW IF EXISTS spo2_pao2_24hr_timediff CASCADE;
create MATERIALIZED VIEW spo2_pao2_24hr_time_diff as
select *, ABS(
             DATE_PART('day', charttime::timestamp  - labtime::timestamp)*24*60
           + DATE_PART('hour', charttime::timestamp - labtime::timestamp)*60
           + DATE_PART('minute', charttime::timestamp - labtime::timestamp)) as time_diff
        from spo2_pao2_24hr
;

DROP MATERIALIZED VIEW IF EXISTS spo2_pao2_24hr_min_timediff CASCADE;
create MATERIALIZED VIEW spo2_pao2_24hr_min_timediff as
select spo2_pao2_24hr_time_diff.*, b.min_time_diff from spo2_pao2_24hr_time_diff join
    (select subject_id, icustay_id, min(time_diff) as min_time_diff from spo2_pao2_24hr_time_diff
        group by subject_id, icustay_id) as b
on spo2_pao2_24hr_time_diff.subject_id = b.subject_id
       and spo2_pao2_24hr_time_diff.icustay_id = b.icustay_id
    and spo2_pao2_24hr_time_diff.time_diff = b.min_time_diff;

-- the charttime and labtime should be as close as possible, it should be controlled in 30 minutes, and delete all the example that has null spo2 value
DROP MATERIALIZED VIEW IF EXISTS spo2_pao2_24hr_min_timediff_unique CASCADE;
create MATERIALIZED VIEW spo2_pao2_24hr_min_timediff_unique as
select spo2_pao2_24hr_min_timediff.* from spo2_pao2_24hr_min_timediff join
    (select subject_id, icustay_id, min(charttime) as charttime from spo2_pao2_24hr_min_timediff
        group by subject_id, icustay_id) as b
on spo2_pao2_24hr_min_timediff.subject_id = b.subject_id
       and spo2_pao2_24hr_min_timediff.icustay_id = b.icustay_id
    and spo2_pao2_24hr_min_timediff.charttime = b.charttime
where min_time_diff <=30 and value is not null;


DROP MATERIALIZED VIEW IF EXISTS spo2_otheriterm CASCADE;
create MATERIALIZED VIEW spo2_otheriterm as
select chartevents.subject_id,chartevents.icustay_id,chartevents.itemid,chartevents.charttime,chartevents.value,chartevents.valueuom,
       s.charttime as pick_charttime,s.starttime,s.endtime
from mimiciii.chartevents
join spo2_pao2_24hr_min_timediff_unique as s on chartevents.icustay_id = s.icustay_id
and chartevents.charttime between starttime and endtime
where chartevents.itemid in (
                  646,220277,                           -- spo2
                  190, 3420, 3422, 223835,              -- Fio2
                  505, 506, 220339,                     -- peep
                  682, 224685,                          -- vt
                  52, 6702,220052,225312,               -- MAP
                  676,677,223762,                        -- Temperature
                  445, 448, 449, 450, 1340, 1486, 1600, 224687   -- Mv (minute volume)
);

DROP MATERIALIZED VIEW IF EXISTS spo2_categorize_item CASCADE;
create MATERIALIZED VIEW spo2_categorize_item as
select *,
      case
        when itemid in(646,220277)                                     then 'Spo2'
        when itemid in(190, 3420, 3422, 223835)                        then 'Fio2'
        when itemid in(505, 506, 220339)                               then 'Peep'
        when itemid in(682, 224685 )                                   then 'Vt'
        when itemid in (52, 6702,220052,225312)                        then 'Map'
        when itemid in (676,677,223762)                                 then 'Temperature'
        when itemid in (445, 448, 449, 450, 1340, 1486, 1600, 224687)  then 'Mv'
      end
from spo2_otheriterm;

DROP MATERIALIZED VIEW IF EXISTS spo2_eachitem_timediff CASCADE;
create MATERIALIZED VIEW spo2_eachitem_timediff as
select spo2_categorize_item.*, ABS(
             DATE_PART('day', spo2_categorize_item.charttime::timestamp  - spo2_categorize_item.pick_charttime::timestamp)*24*60
           + DATE_PART('hour', spo2_categorize_item.charttime::timestamp - spo2_categorize_item.pick_charttime::timestamp)*60
           + DATE_PART('minute', spo2_categorize_item.charttime::timestamp - spo2_categorize_item.pick_charttime::timestamp)) as time_diff
from spo2_categorize_item;

DROP MATERIALIZED VIEW IF EXISTS spo2_eachitem_min_timediff CASCADE;
create MATERIALIZED VIEW spo2_eachitem_min_timediff as
select spo2_eachitem_timediff.subject_id,spo2_eachitem_timediff.icustay_id,itemid,spo2_eachitem_timediff."case",value,valueuom,charttime,pick_charttime,starttime,endtime,n.min_time_diff
from spo2_eachitem_timediff
join
(select subject_id,icustay_id,"case",min(time_diff) as min_time_diff from spo2_eachitem_timediff
group by subject_id, icustay_id, "case") as n
on spo2_eachitem_timediff.subject_id = n.subject_id
   and spo2_eachitem_timediff.icustay_id = n.icustay_id
   and spo2_eachitem_timediff."case" = n."case"
   and spo2_eachitem_timediff.time_diff = n.min_time_diff;

DROP MATERIALIZED VIEW IF EXISTS spo2_eachitem_min_timediff_unique CASCADE;
create MATERIALIZED VIEW spo2_eachitem_min_timediff_unique as
select distinct spo2_eachitem_min_timediff.*
from spo2_eachitem_min_timediff
join
(select subject_id,icustay_id,"case",min(charttime) as min_charttime
    from spo2_eachitem_min_timediff
    group by subject_id, icustay_id, "case") as m
on spo2_eachitem_min_timediff.subject_id = m.subject_id
   and spo2_eachitem_min_timediff.icustay_id = m.icustay_id
   and spo2_eachitem_min_timediff."case" = m."case"
   and spo2_eachitem_min_timediff.charttime = m.min_charttime
where min_time_diff <= 120 and value is not null;


DROP MATERIALIZED VIEW IF EXISTS pao2_otheriterm CASCADE;
create MATERIALIZED VIEW pao2_otheriterm as
select labevents.subject_id,labevents.itemid,labevents.charttime,labevents.value,labevents.valueuom,
       s.labtime as pick_labtime,s.starttime,s.endtime
from mimiciii.labevents
join spo2_pao2_24hr_min_timediff_unique as s on labevents.subject_id = s.subject_id
and labevents.charttime between starttime and endtime
where labevents.itemid in (
                  50821,490,                 -- pao2
                  50818,                     -- paco2
                  50817,                     -- sao2
                  50831,50820,               -- ph
                  50811,                     -- hemoglobin
                  50804                     -- Total co2
);
DROP MATERIALIZED VIEW IF EXISTS pao2_categorize_item CASCADE;
create MATERIALIZED VIEW pao2_categorize_item as
select *,
      case
        when itemid in(50821,490)              then 'Pao2'
        when itemid in(50818)                  then 'Paco2'
        when itemid in(50817)                  then 'Sao2'
        when itemid in(50831,50820)            then 'Ph'
        when itemid in(50811)                  then 'Hemoglobin'
        when itemid in (50804)                 then 'Co2'
      end
from pao2_otheriterm;
DROP MATERIALIZED VIEW IF EXISTS pao2_eachitem_timediff CASCADE;
create MATERIALIZED VIEW pao2_eachitem_timediff as
select pao2_categorize_item.*, ABS(
             DATE_PART('day', pao2_categorize_item.charttime::timestamp  - pao2_categorize_item.pick_labtime::timestamp)*24*60
           + DATE_PART('hour', pao2_categorize_item.charttime::timestamp - pao2_categorize_item.pick_labtime::timestamp)*60
           + DATE_PART('minute', pao2_categorize_item.charttime::timestamp - pao2_categorize_item.pick_labtime::timestamp)) as time_diff
from pao2_categorize_item;
DROP MATERIALIZED VIEW IF EXISTS pao2_eachitem_min_timediff CASCADE;
create MATERIALIZED VIEW pao2_eachitem_min_timediff as
select distinct pao2_eachitem_timediff.subject_id,pao2_eachitem_timediff.itemid,pao2_eachitem_timediff."case",value,valueuom,charttime,pick_labtime,starttime,endtime,n.min_time_diff
from pao2_eachitem_timediff
join
(select subject_id,"case",min(time_diff) as min_time_diff from pao2_eachitem_timediff
group by subject_id, "case") as n
on pao2_eachitem_timediff.subject_id = n.subject_id
   and pao2_eachitem_timediff."case" = n."case"
   and pao2_eachitem_timediff.time_diff = n.min_time_diff
where (n."case" <> 'Hemoglobin' and min_time_diff = 0) or (n."case" = 'Hemoglobin' and min_time_diff <= 1440);



DROP MATERIALIZED VIEW IF EXISTS spo2_pao2 CASCADE;
create MATERIALIZED VIEW spo2_pao2 as
select subject_id,itemid,"case",value,valueuom,charttime,pick_charttime,starttime,endtime,min_time_diff
from spo2_eachitem_min_timediff_unique
union
select subject_id,itemid,"case",value,valueuom,charttime,pick_labtime,starttime,endtime,min_time_diff
from pao2_eachitem_min_timediff

-- vasopressor duration
DROP MATERIALIZED VIEW IF EXISTS VASOPRESSORDURATIONS;
CREATE MATERIALIZED VIEW VASOPRESSORDURATIONS as
with io_cv as
(
  select
    icustay_id, charttime, itemid, stopped
    -- ITEMIDs (42273, 42802) accidentally store rate in amount column
    , case
        when itemid in (42273, 42802)
          then amount
        else rate
      end as rate
    , case
        when itemid in (42273, 42802)
          then rate
        else amount
      end as amount
  from mimiciii.inputevents_cv
  where itemid in
  (
    30047,30120,30044,30119,30309,30127
  , 30128,30051,30043,30307,30042,30306,30125
  , 42273, 42802
  )
)
-- select only the ITEMIDs from the inputevents_mv table related to vasopressors
, io_mv as
(
  select
    icustay_id, linkorderid, starttime, endtime
  from mimiciii.inputevents_mv io
  -- Subselect the vasopressor ITEMIDs
  where itemid in
  (
  221906,221289,221749,222315,221662,221653,221986
  )
  and statusdescription != 'Rewritten' -- only valid orders
)
, vasocv1 as
(
  select
    icustay_id, charttime, itemid
    -- case statement determining whether the ITEMID is an instance of vasopressor usage
    , 1 as vaso

    -- the 'stopped' column indicates if a vasopressor has been disconnected
    , max(case when stopped in ('Stopped','D/C''d') then 1
          else 0 end) as vaso_stopped

    , max(case when rate is not null then 1 else 0 end) as vaso_null
    , max(rate) as vaso_rate
    , max(amount) as vaso_amount

  from io_cv
  group by icustay_id, charttime, itemid
)
, vasocv2 as
(
  select v.*
    , sum(vaso_null) over (partition by icustay_id, itemid order by charttime) as vaso_partition
  from
    vasocv1 v
)
, vasocv3 as
(
  select v.*
    , first_value(vaso_rate) over (partition by icustay_id, itemid, vaso_partition order by charttime) as vaso_prevrate_ifnull
  from
    vasocv2 v
)
, vasocv4 as
(
select
    icustay_id
    , charttime
    , itemid
    -- , (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, vaso order by charttime))) AS delta

    , vaso
    , vaso_rate
    , vaso_amount
    , vaso_stopped
    , vaso_prevrate_ifnull

    -- We define start time here
    , case
        when vaso = 0 then null

        -- if this is the first instance of the vasoactive drug
        when vaso_rate > 0 and
          LAG(vaso_prevrate_ifnull,1)
          OVER
          (
          partition by icustay_id, itemid, vaso, vaso_null
          order by charttime
          )
          is null
          then 1

        -- you often get a string of 0s
        -- we decide not to set these as 1, just because it makes vasonum sequential
        when vaso_rate = 0 and
          LAG(vaso_prevrate_ifnull,1)
          OVER
          (
          partition by icustay_id, itemid, vaso
          order by charttime
          )
          = 0
          then 0

        -- sometimes you get a string of NULL, associated with 0 volumes
        -- same reason as before, we decide not to set these as 1
        -- vaso_prevrate_ifnull is equal to the previous value *iff* the current value is null
        when vaso_prevrate_ifnull = 0 and
          LAG(vaso_prevrate_ifnull,1)
          OVER
          (
          partition by icustay_id, itemid, vaso
          order by charttime
          )
          = 0
          then 0

        -- If the last recorded rate was 0, newvaso = 1
        when LAG(vaso_prevrate_ifnull,1)
          OVER
          (
          partition by icustay_id, itemid, vaso
          order by charttime
          ) = 0
          then 1

        -- If the last recorded vaso was D/C'd, newvaso = 1
        when
          LAG(vaso_stopped,1)
          OVER
          (
          partition by icustay_id, itemid, vaso
          order by charttime
          )
          = 1 then 1

        -- ** not sure if the below is needed
        --when (CHARTTIME - (LAG(CHARTTIME, 1) OVER (partition by icustay_id, vaso order by charttime))) > (interval '4 hours') then 1
      else null
      end as vaso_start

FROM
  vasocv3
)
-- propagate start/stop flags forward in time
, vasocv5 as
(
  select v.*
    , SUM(vaso_start) OVER (partition by icustay_id, itemid, vaso order by charttime) as vaso_first
FROM
  vasocv4 v
)
, vasocv6 as
(
  select v.*
    -- We define end time here
    , case
        when vaso = 0
          then null

        -- If the recorded vaso was D/C'd, this is an end time
        when vaso_stopped = 1
          then vaso_first

        -- If the rate is zero, this is the end time
        when vaso_rate = 0
          then vaso_first

        -- the last row in the table is always a potential end time
        -- this captures patients who die/are discharged while on vasopressors
        -- in principle, this could add an extra end time for the vasopressor
        -- however, since we later group on vaso_start, any extra end times are ignored
        when LEAD(CHARTTIME,1)
          OVER
          (
          partition by icustay_id, itemid, vaso
          order by charttime
          ) is null
          then vaso_first

        else null
        end as vaso_stop
    from vasocv5 v
)

-- -- if you want to look at the results of the table before grouping:
-- select
--   icustay_id, charttime, vaso, vaso_rate, vaso_amount
--     , case when vaso_stopped = 1 then 'Y' else '' end as stopped
--     , vaso_start
--     , vaso_first
--     , vaso_stop
-- from vasocv6 order by charttime;


, vasocv as
(
-- below groups together vasopressor administrations into groups
select
  icustay_id
  , itemid
  -- the first non-null rate is considered the starttime
  , min(case when vaso_rate is not null then charttime else null end) as starttime
  -- the *first* time the first/last flags agree is the stop time for this duration
  , min(case when vaso_first = vaso_stop then charttime else null end) as endtime
from vasocv6
where
  vaso_first is not null -- bogus data
and
  vaso_first != 0 -- sometimes *only* a rate of 0 appears, i.e. the drug is never actually delivered
and
  icustay_id is not null -- there are data for "floating" admissions, we don't worry about these
group by icustay_id, itemid, vaso_first
having -- ensure start time is not the same as end time
 min(charttime) != min(case when vaso_first = vaso_stop then charttime else null end)
and
  max(vaso_rate) > 0 -- if the rate was always 0 or null, we consider it not a real drug delivery
)
-- we do not group by ITEMID in below query
-- this is because we want to collapse all vasopressors together
, vasocv_grp as
(
SELECT
  s1.icustay_id,
  s1.starttime,
  MIN(t1.endtime) AS endtime
FROM vasocv s1
INNER JOIN vasocv t1
  ON  s1.icustay_id = t1.icustay_id
  AND s1.starttime <= t1.endtime
  AND NOT EXISTS(SELECT * FROM vasocv t2
                 WHERE t1.icustay_id = t2.icustay_id
                 AND t1.endtime >= t2.starttime
                 AND t1.endtime < t2.endtime)
WHERE NOT EXISTS(SELECT * FROM vasocv s2
                 WHERE s1.icustay_id = s2.icustay_id
                 AND s1.starttime > s2.starttime
                 AND s1.starttime <= s2.endtime)
GROUP BY s1.icustay_id, s1.starttime
ORDER BY s1.icustay_id, s1.starttime
)
-- now we extract the associated data for metavision patients
-- do not need to group by itemid because we group by linkorderid
, vasomv as
(
  select
    icustay_id, linkorderid
    , min(starttime) as starttime, max(endtime) as endtime
  from io_mv
  group by icustay_id, linkorderid
)
, vasomv_grp as
(
SELECT
  s1.icustay_id,
  s1.starttime,
  MIN(t1.endtime) AS endtime
FROM vasomv s1
INNER JOIN vasomv t1
  ON  s1.icustay_id = t1.icustay_id
  AND s1.starttime <= t1.endtime
  AND NOT EXISTS(SELECT * FROM vasomv t2
                 WHERE t1.icustay_id = t2.icustay_id
                 AND t1.endtime >= t2.starttime
                 AND t1.endtime < t2.endtime)
WHERE NOT EXISTS(SELECT * FROM vasomv s2
                 WHERE s1.icustay_id = s2.icustay_id
                 AND s1.starttime > s2.starttime
                 AND s1.starttime <= s2.endtime)
GROUP BY s1.icustay_id, s1.starttime
ORDER BY s1.icustay_id, s1.starttime
)
select
  icustay_id
  -- generate a sequential integer for convenience
  , ROW_NUMBER() over (partition by icustay_id order by starttime) as vasonum
  , starttime, endtime
  , extract(epoch from endtime - starttime)/60/60 AS duration_hours
  -- add durations
from
  vasocv_grp

UNION

select
  icustay_id
  , ROW_NUMBER() over (partition by icustay_id order by starttime) as vasonum
  , starttime, endtime
  , extract(epoch from endtime - starttime)/60/60 AS duration_hours
  -- add durations
from
  vasomv_grp

order by icustay_id, vasonum;
-- vasopressor duration end


DROP MATERIALIZED VIEW IF EXISTS Age_gender;
CREATE MATERIALIZED VIEW Age_gender as
select p.* ,q.dob,q.gender,
round((cast(p.charttime as date)-CAST(q.dob as date))/365.242,0) as age
from spo2_eachitem_min_timediff_unique as p
inner join mimiciii.patients as q
on q.subject_id= p.subject_id;

DROP MATERIALIZED VIEW IF EXISTS Age_gender_veso;
CREATE MATERIALIZED VIEW Age_gender_veso as
select m.*,n.vasonum,n.starttime as veso_start,n.endtime as veso_end,
    case
        when  m.pick_charttime between n.starttime and n.endtime
            then '1'
        else '0'
    end as Vesopression_infussion
from age_gender as m
inner join vasopressordurations n
on m.icustay_id = n.icustay_id;






