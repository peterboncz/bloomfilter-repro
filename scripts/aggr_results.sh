#!/bin/bash

#=----------------------------------------------------------------------------=#
# This script consumes the output produced by 'benchmark.sh' and extracts the
# data points that are required to produce a summary plot similar to the one
# in the paper on page 10.
#=----------------------------------------------------------------------------=#

DATADIR="./results"
CSVFILE="$DATADIR/skyline_data.csv"
DBFILE="${DATADIR}/skyline.sqlite3"
OUTDIR="./plot"

#LOAD_MATH="SELECT load_extension('libsqlitefunctions');"
export LD_PRELOAD=/lib/x86_64-linux-gnu/libm.so.6
SQL="sqlite3 $DBFILE"

mkdir $OUTDIR

echo "parsing the raw data..."
cat $DATADIR/results_*.out | grep -e "^\"{\"\"name\"\":" > $CSVFILE

echo "reading $CSVFILE..."
# a table for the raw results
$SQL "drop table if exists raw_results;"
$SQL "CREATE TABLE raw_results(
    filter json,
    m int,
    b real,
    n int,
    s real,
    insert_time_nanos int,
    false_positives int,
    fpr real,
    lookups_per_sec real,
    cycles_per_lookup real,
    thread_cnt int,
    scalar_code int
    );"
# Import the data
echo ".mode csv
.import $CSVFILE raw_results" | $SQL

# Clean data
echo "data cleaning..."
$SQL "update raw_results set filter = json_insert(filter,'$.addr', 'pow2');" # older results do not have the 'addr' field. in that case 'addr' = 'pow2'
$SQL "update raw_results set filter = json_insert(filter,'$.z', 1);" # older results do not have the 'z' (zone count) field. in that case 'z' = 1
$SQL "delete from raw_results where json_extract(filter, '$.word_size') * json_extract(filter, '$.w') > 64;" # ignore BBFs that exceed a single cache line
$SQL "update raw_results set filter = json_remove(filter,'$.delete_support') where json_extract(filter, '$.name') = 'cuckoo';"


if [ ! -z $1 ]; then
    echo "number of threads: $1"
    $SQL "delete from raw_results where n = 0 and thread_cnt <> $1;"
fi

$SQL -column -header "select thread_cnt, count(*) data_points from raw_results where n = 0 group by thread_cnt;"

tmpfile=`mktemp`
echo "select thread_cnt, count(*) from raw_results where n = 0 group by thread_cnt;" | $SQL > $tmpfile
outputsize=`cat $tmpfile | wc -l`
if [ -z $1 ] && [[ $outputsize != 1 ]]
then
    echo ""
    echo -n "The raw data contains multiple experimental results using"
    echo -n " different concurrency levels (numbers of threads)."
    echo -n " These results are not comparable and therefore need to be analyzed"
    echo    " separately."
    echo -n "Please re-run the script and specify the result set of interest as"
    echo    " follows:"
    echo    " $0 <NUMBER_OF_THREADS>"
    exit 1
fi
rm $tmpfile
datapoints=`$SQL "select count(*) from raw_results where n = 0;"`
if [[ $datapoints == 0 ]]; then
    echo ""
    echo "No (matching) data points found. Aborting."
    exit 1
else
    echo "number of data points: $datapoints"
fi


echo "populating tables with n and tw values (of interest)..."
# Values for n
$SQL "drop table if exists n_values"
q=1.44
#q=0
if [[ $q == 0 ]]; then
$SQL "create table n_values as
        select distinct n as n from raw_results where n > 0;"
else
  t=`mktemp`
  Q=$q ./n_filter > $t
  $SQL "create table n_values (n int);"
  echo "
.mode csv
.import $t n_values" | $SQL
  $SQL "insert into n_values values (0);" # keep performance results (n=0)
fi
n_s=`$SQL "select n from n_values order by n;"`


echo -n "values for n are: "
echo  $n_s
n_cnt=`echo "$n_s" | wc -l`
echo "considering $n_cnt different values for n."
# Delete data points with invalid n's
echo "filtering raw data on 'n'..."
$SQL "delete from raw_results
 where not exists (select n from n_values where n=raw_results.n);"

# adjust y scale for plot
#yscale=`echo "scale=5; 0.0305/$n_cnt" |bc`
xscale="0.65"
yscale="0.85"


# Values for t_w
$SQL "drop table if exists tw_values;"
$SQL "create table tw_values (tw int unique);"
#$SQL "WITH RECURSIVE cnt(x) AS (VALUES(8) UNION ALL SELECT x*2 FROM cnt WHERE x<10000000000) insert into tw_values SELECT x FROM cnt;"
#$SQL "WITH RECURSIVE cnt(x) AS (VALUES(8) UNION ALL SELECT x*2 FROM cnt WHERE x<4294967296) insert into tw_values SELECT x FROM cnt;"
$SQL "WITH RECURSIVE cnt(x) AS (VALUES(16) UNION ALL SELECT x*2 FROM cnt WHERE  x<2147483648) insert into tw_values SELECT x FROM cnt;"
tw_s=`$SQL "select distinct tw from tw_values order by tw;"`

# Filters
#filters=`$SQL "select distinct json_extract(filter, '$.name') as filter_name from raw_results order by filter_name;"`
filters="blocked_bloom_multiword
blocked_bloom_impala
blocked_cuckoo
cuckoo"
#echo "filters: $filters"

#--------------------------------------------
# Aggregate the results of the different runs
#--------------------------------------------
# Performance results
# Notes:
#  - scalar_code is deprecated
#  - performance measurements can be identified with n=0
#  - WARNING: remove the unrolling attribute u before group by filter!
echo "aggregating performance data over the independent runs..."
$SQL "drop table if exists perf_results;"
$SQL "create table perf_results as
  select json_remove(filter,'$.u') as filter,
         json_extract(filter, '$.name') as filter_name,
         m, thread_cnt,
         max(lookups_per_sec) as lookups_per_sec,
         min(cycles_per_lookup) as cycles_per_lookup
    from raw_results r
   where n=0 and scalar_code=0 -- during performance experiments, no elements are inserted into the filter n=0
   group by json_remove(filter,'$.u'), m, thread_cnt;"

# Precision results
echo "aggregating precision data over the independent runs..."
$SQL "drop table if exists prec_results;"
$SQL "create table prec_results as
    select json_remove(filter,'$.u') as filter,
           json_extract(filter, '$.name') as filter_name,
           m, n, (m*1.0)/n as b,
           avg(false_positives) as false_positives,
           avg(fpr) as fpr,
           count(*) as num_data_points
      from raw_results
     --where n>0 and b<=32 and fpr>0
     where n>0 -- precision experiments require at least one inserted element: n > 0
--     and false_positives>100
--     and b<=32
--     and fpr>0
     group by json_remove(filter,'$.u'), m, n;"



echo "joining performance and precision results..."
# combined performance and precision results
# Note: only works for branch-free algorithms, where the costs for positive and negative queries are equal
$SQL "drop table if exists results;"
$SQL "create table results as
    select prec.*, perf.lookups_per_sec, perf.cycles_per_lookup
      from prec_results prec, perf_results perf
     where perf.filter=prec.filter and perf.m=prec.m;"

# add memory restriction (20 bits per element).
$SQL "delete from results where b > 20;"


#$SQL "create view results as
#  select json_remove(filter,'$.u') as filter,
#         json_extract(filter, '$.name') as filter_name,
#         m, (m*1.0)/n as b, n, avg(false_positives) as false_positives,
#         avg(fpr) as fpr, min(cycles_per_lookup) as cycles_per_lookup
#    from raw_results
#   where thread_cnt = 1 -- SINGLE THREADED
#     and fpr > 0
#     and b <= 32
#--     and n <= 67108864
#     and n <= 16777216
#
#   group by filter, m n;"

echo "indexing results..."
$SQL "create index n_idx on results (n);"
echo "computing skyline..."
$SQL "drop table if exists skyline;"
$SQL "create table skyline as
select n_values.n, tw_values.tw,
  (select filter from
     (select filter, (r.cycles_per_lookup + r.fpr * tw_values.tw) as overhead from results r where r.n = n_values.n
      order by overhead limit 1)
  ) as filter
from n_values, tw_values;"

# Create / empty plot file
PLOTFILE="${OUTDIR}/skyline_filter.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

plotcolor=0

for f in $filters; do
plotcolor=$((plotcolor + 1))

tmpfile=`mktemp`
echo ".mode tab
select tw, n, 'def' from skyline where json_extract(filter, '$.name') = '$f' order by tw, n;" | $SQL > $tmpfile
outputsize=`cat $tmpfile | wc -l`
if [[ $outputsize != 0 ]]; then
echo "
% Filter: '$f'
\addplot[
  scatter,
  only marks,
  point meta=explicit symbolic,
  scatter/classes={
    def={mark=square*,xscale=$xscale,yscale=$yscale,filtercolor$plotcolor}
  }%
]
table[meta=label] {
n tw label
" >> $PLOTFILE

cat $tmpfile >> $PLOTFILE

f_name="n/a"
if [[ $f == *"impala"* ]]; then
    f_name="Impala"
elif [[ $f == *"multiword"* ]]; then
    f_name="Blocked Bloom"
elif [[ $f == "blocked_cuckoo" ]]; then
    f_name="Blocked Cuckoo"
elif [[ $f == "cuckoo" ]]; then
    f_name="Cuckoo"
fi
echo "
}; % end of table
\addlegendentry{$f_name}
" | sed 's/_/ /g' >> $PLOTFILE
fi
rm $tmpfile

done # for each filter


#=----------------------------------------------------------------------------=#
#
# Blocked Bloom Skyline
#
#=----------------------------------------------------------------------------=#
echo "extracting blocked bloom filter results..."
$SQL "drop table if exists bbf;"
$SQL "create table bbf as
select
  json_extract(filter, '$.word_size') * json_extract(filter, '$.w') as block_size,
  json_extract(filter, '$.word_size') as word_size,
  json_extract(filter, '$.w') as word_cnt,
  json_extract(filter, '$.s') as sector_cnt,
  json_extract(filter, '$.z') as zone_cnt,
  (json_extract(filter, '$.word_size') * json_extract(filter, '$.w')) / json_extract(filter, '$.s') as sector_size,
  json_extract(filter, '$.k') as k,
  json_extract(filter, '$.addr') as addr,
  case when json_extract(filter, '$.w') <= json_extract(filter, '$.s')
    then
        case when json_extract(filter, '$.w') = 1
        then 'single'
        else
         case when json_extract(filter, '$.z') > 1
         then 'zoned'
         else 'seq'
         end
        end
    else 'rnd'
  end as access,
  m, b, n, false_positives, fpr, cycles_per_lookup, filter
 from results
where json_extract(filter, '$.name') = 'blocked_bloom_multiword';
"
$SQL "create index n_idx_bbf on bbf (n);"


echo "computing blocked bloom filter skyline..."
$SQL "drop table if exists bbf_skyline;"
$SQL "create table bbf_skyline as
select n_values.n, tw_values.tw,
  (select filter_info from
    (select json_object('block_size',bbf.block_size,
                        'sector_size',bbf.sector_size,
                        'word_size',bbf.word_size,
                        'w', bbf.word_cnt,
                        's', bbf.sector_cnt,
                        'z', bbf.zone_cnt,
                        'k', bbf.k,
                        'access', bbf.access,
                        'addr', bbf.addr,
                        'overhead', (bbf.cycles_per_lookup + bbf.fpr * tw_values.tw),
                        'size', json_extract(bbf.filter, '$.size')
                        ) as filter_info,
            (bbf.cycles_per_lookup + bbf.fpr * tw_values.tw) as overhead
       from bbf
      where bbf.n = n_values.n
      order by overhead limit 1
    )
  ) as filter_info
 from n_values, tw_values;"


echo "determining block sizes..."
bs=`$SQL "select distinct json_extract(filter_info, '$.block_size') as block_size from bbf_skyline order by block_size;"`
echo "block sizes:"
echo $bs


#=----------------------------------------------------------------------------=#
# Block size
#=----------------------------------------------------------------------------=#
# Create / empty plot file
PLOTFILE="${OUTDIR}/skyline_bbf_blocksize.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

plotcolor=0
for b in $bs; do
plotcolor=$((plotcolor + 1))

tmpfile=`mktemp`
echo ".mode tab
select tw, n, 'def' from bbf_skyline where json_extract(filter_info, '$.block_size') = $b order by tw, n;" | $SQL > $tmpfile
outputsize=`cat $tmpfile | wc -l`
if [[ $outputsize != 0 ]]; then
echo "
% block size: '$b'
\addplot[
  scatter,
  only marks,
  point meta=explicit symbolic,
  scatter/classes={
    def={mark=square*,xscale=$xscale,yscale=$yscale,blocksizecolor$plotcolor}
  }%
]
table[meta=label] {
tw n label
" >> $PLOTFILE

cat $tmpfile >> $PLOTFILE

echo "
}; % end of table
\addlegendentry{$b Bytes\,\,}
" | sed 's/_/ /g' >> $PLOTFILE
fi
rm $tmpfile
done


#=----------------------------------------------------------------------------=#
# Block Access Patterns
#=----------------------------------------------------------------------------=#
# Create / empty plot file
PLOTFILE="${OUTDIR}/skyline_bbf_access_pattern.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

#a_s=`$SQL "select distinct json_extract(filter_info, '$.access') as info from bbf_skyline order by info;"`
a_s="rnd
seq
zoned
single"

# do not distinguish between sequential and zoned
#a_s=`$SQL "select distinct case when json_extract(filter_info, '$.access') = 'zoned' then 'seq' else json_extract(filter_info, '$.access')  end as info from bbf_skyline order by info;"`
echo "access patterns are: $a_s"

plotcolor=0
for a in $a_s; do
plotcolor=$((plotcolor + 1))

tmpfile=`mktemp`
echo ".mode tab
select tw, n, 'def' from bbf_skyline where json_extract(filter_info, '$.access') = '$a' order by tw, n;" | $SQL >> $tmpfile
#echo ".mode tab
#select tw, n, 'def' from bbf_skyline where case when json_extract(filter_info, '$.access') = 'zoned' then 'seq' else json_extract(filter_info, '$.access') end = '$a' order by tw, n;" | $SQL >> $tmpfile
outputsize=`cat $tmpfile | wc -l`

if [[ $outputsize != 0 ]]; then
echo "
% access: '$a'
\addplot[
  scatter,
  only marks,
  point meta=explicit symbolic,
  scatter/classes={
    def={mark=square*,xscale=$xscale,yscale=$yscale,accesscolor$plotcolor}
  }%
]
table[meta=label] {
tw n label
" >> $PLOTFILE

cat $tmpfile >> $PLOTFILE

a_name="n/a"
if [[ $a == "seq" ]]; then
#    a_name="Sequential"
    a_name="Sectorized"
elif [[ $a == "rnd" ]]; then
    a_name="Blocked"
#    a_name="Random"
elif [[ $a == "single" ]]; then
    a_name="Register-blocked"
#    a_name="Single word"
elif [[ $a == "zoned" ]]; then
    a_name="Cache-sectorized"
#    a_name="Sequential/Zoned"
fi
echo "
}; % end of table
\addlegendentry{$a_name}
" | sed 's/_/ /g' >> $PLOTFILE

fi
rm $tmpfile
done


#=----------------------------------------------------------------------------=#
# Sector count
#=----------------------------------------------------------------------------=#
# Create / empty plot file
PLOTFILE="${OUTDIR}/skyline_bbf_sector_cnt.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

sc=`$SQL "select distinct json_extract(filter_info, '$.s') as sector_cnt from bbf_skyline order by sector_cnt;"`

plotcolor=0
for s in $sc; do
plotcolor=$((plotcolor + 1))

tmpfile=`mktemp`
echo ".mode tab
select tw, n, 'def' from bbf_skyline where json_extract(filter_info, '$.s') = $s order by tw, n;" | $SQL >> $tmpfile
outputsize=`cat $tmpfile | wc -l`
if [[ $outputsize != 0 ]]; then
echo "
% sector cnt: '$s'
\addplot[
  scatter,
  only marks,
  point meta=explicit symbolic,
  scatter/classes={
    def={mark=square*,xscale=$xscale,yscale=$yscale,sectorcntcolor$plotcolor}
  }%
]
table[meta=label] {
tw n label
" >> $PLOTFILE

cat $tmpfile >> $PLOTFILE

echo "
}; % end of table
\addlegendentry{$s\,\,}
" | sed 's/_/ /g' >> $PLOTFILE
fi
rm $tmpfile
done


#=----------------------------------------------------------------------------=#
# Zone count
#=----------------------------------------------------------------------------=#
# Create / empty plot file
PLOTFILE="${OUTDIR}/skyline_bbf_zone_cnt.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

zc=`$SQL "select distinct json_extract(filter_info, '$.z') as zone_cnt from bbf_skyline order by zone_cnt;"`

plotcolor=0
for z in $zc; do
plotcolor=$((plotcolor + 1))

tmpfile=`mktemp`
echo ".mode tab
select tw, n, 'def' from bbf_skyline where json_extract(filter_info, '$.z') = $z order by tw, n;" | $SQL >> $tmpfile
outputsize=`cat $tmpfile | wc -l`
if [[ $outputsize != 0 ]]; then
echo "
% zone cnt: '$z'
\addplot[
  scatter,
  only marks,
  point meta=explicit symbolic,
  scatter/classes={
    def={mark=square*,xscale=$xscale,yscale=$yscale,zonecntcolor$plotcolor}
  }%
]
table[meta=label] {
tw n label
" >> $PLOTFILE

cat $tmpfile >> $PLOTFILE

echo "
}; % end of table
\addlegendentry{\$z=$z\$\,\,}
" | sed 's/_/ /g' >> $PLOTFILE
fi
rm $tmpfile
done


#=----------------------------------------------------------------------------=#
# Word size
#=----------------------------------------------------------------------------=#
# Create / empty plot file
PLOTFILE="${OUTDIR}/skyline_bbf_word_size.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

ws=`$SQL "select distinct json_extract(filter_info, '$.word_size') as word_size from bbf_skyline order by word_size;"`

plotcolor=0
for w in $ws; do
plotcolor=$((plotcolor + 1))

tmpfile=`mktemp`
echo ".mode tab
select tw, n, 'def' from bbf_skyline where json_extract(filter_info, '$.word_size') = $w order by tw, n;" | $SQL >> $tmpfile
outputsize=`cat $tmpfile | wc -l`
if [[ $outputsize != 0 ]]; then
echo "
% word size: '$w'
\addplot[
  scatter,
  only marks,
  point meta=explicit symbolic,
  scatter/classes={
    def={mark=square*,xscale=$xscale,yscale=$yscale,wordcntcolor$plotcolor}
  }%
]
table[meta=label] {
tw n label
" >> $PLOTFILE

cat $tmpfile >> $PLOTFILE

echo "
}; % end of table
\addlegendentry{$w Bytes\,\,}
" | sed 's/_/ /g' >> $PLOTFILE
fi
rm $tmpfile
done


#=----------------------------------------------------------------------------=#
# K
#=----------------------------------------------------------------------------=#
# Create / empty plot file
PLOTFILE="${OUTDIR}/skyline_bbf_k.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

k_s=`$SQL "select distinct json_extract(filter_info, '$.k') as info from bbf_skyline order by info;"`

for k in $k_s; do
echo "
% k: '$k'
\addplot[
  scatter,
  only marks,
  point meta=explicit symbolic,
  scatter/classes={
    def={mark=square*,xscale=$xscale,yscale=$yscale,kcolor$k}
  }%
]
table[meta=label] {
tw n label
" >> $PLOTFILE

echo ".mode tab
select tw, n, 'def' from bbf_skyline where json_extract(filter_info, '$.k') = $k order by tw, n;" | $SQL >> $PLOTFILE

echo "
}; % end of table
\addlegendentry{$k}
" >> $PLOTFILE

done


#=----------------------------------------------------------------------------=#
# Addressing modes
#=----------------------------------------------------------------------------=#
# Create / empty plot file
PLOTFILE="${OUTDIR}/skyline_bbf_addr.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

#addr_s=`$SQL "select distinct json_extract(filter_info, '$.addr') as info from bbf_skyline order by info;"`

plotcolor=0
#for addr in $addr_s; do
for addr in pow2 magic; do
plotcolor=$((plotcolor + 1))

tmpfile=`mktemp`
echo ".mode tab
select tw, n, 'def' from bbf_skyline where json_extract(filter_info, '$.addr') = '$addr' order by tw, n;" | $SQL >> $tmpfile
outputsize=`cat $tmpfile | wc -l`
if [[ $outputsize != 0 ]]; then
echo "
% addr: '$addr'
\addplot[
  scatter,
  only marks,
  point meta=explicit symbolic,
  scatter/classes={
    def={mark=square*,xscale=$xscale,yscale=$yscale,addrcolor$plotcolor}
  }%
]
table[meta=label] {
tw n label
" >> $PLOTFILE

cat $tmpfile >> $PLOTFILE

addr_name="n/a"
if [[ $addr == "pow2" ]]; then
    addr_name="Power of two"
elif [[ $addr == "magic" ]]; then
    addr_name="Magic"
fi
echo "
}; % end of table
\addlegendentry{$addr_name}
" >> $PLOTFILE
fi
rm $tmpfile
done


#=----------------------------------------------------------------------------=#
#
# Cuckoo Skyline
#
#=----------------------------------------------------------------------------=#
echo "extracting cuckoo filter results..."
$SQL "drop table if exists cf;"
$SQL "drop index if exists n_idx_cf;"
$SQL "create table cf as
select
  json_extract(filter, '$.tag_bits') as tag_bits,
  json_extract(filter, '$.associativity') as associativity,
  json_extract(filter, '$.addr') as addr,
  m, b, n, false_positives, fpr,cycles_per_lookup, filter
 from results
where json_extract(filter, '$.name') = 'cuckoo';
"
echo "indexing cuckoo filter results..."
$SQL "create index n_idx_cf on cf (n);"


echo "computing cuckoo filter skyline..."
$SQL "drop table if exists cf_skyline;"
$SQL "create table cf_skyline as
select n_values.n, tw_values.tw,
  (select filter_info from
    (select json_object('tag_bits', cf.tag_bits,
                        'associativity', cf.associativity,
                        'overhead', (cf.cycles_per_lookup + cf.fpr * tw_values.tw),
                        'size', json_extract(filter, '$.size'),
                        'addr', cf.addr
                        ) as filter_info,
            (cf.cycles_per_lookup + cf.fpr * tw_values.tw) as overhead
       from cf
      where cf.n = n_values.n
      order by overhead limit 1
    )
  ) as filter_info
 from n_values, tw_values;"


#=----------------------------------------------------------------------------=#
# Compute the performance differences (BBF vs CF)
#=----------------------------------------------------------------------------=#
echo "computing relative speedups (cuckoo vs. blocked bloom)..."
$SQL "drop table if exists perf_diff;"
$SQL "create table perf_diff as
select bs.n as n, bs.tw as tw,
  case
    when json_extract(bs.filter_info, '$.overhead') < json_extract(cs.filter_info, '$.overhead')
    then 'bbf'
    else 'cf'
  end as filter_class,
  case
    when json_extract(bs.filter_info, '$.overhead') < json_extract(cs.filter_info, '$.overhead')
    then json_extract(cs.filter_info, '$.overhead') / json_extract(bs.filter_info, '$.overhead')
    else json_extract(bs.filter_info, '$.overhead') / json_extract(cs.filter_info, '$.overhead')
  end as speedup
   from bbf_skyline bs, cf_skyline cs
  where bs.n=cs.n and bs.tw=cs.tw
  order by n, tw;
"

PLOTFILE="${OUTDIR}/skyline_bbf_cf_performance_diff.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

inf=99999999
declare -a speedup_steps=(1 1.05 1.1 1.25 1.5 1.75 2 3 4 5 10 $inf)

len=${#speedup_steps[@]}

for ((i=1; i<${len}; i++)); do

begin_idx=$((i-1))
end_idx=i
begin=${speedup_steps[$begin_idx]}
end=${speedup_steps[$end_idx]}

tmpfile=`mktemp`
echo ".mode tab
select tw, n, 'def' from perf_diff where speedup >= $begin and speedup < $end order by tw, n;" | $SQL >> $tmpfile
outputsize=`cat $tmpfile | wc -l`
if [[ $outputsize != 0 ]]; then
echo "
% s: '[$begin, $end)'
\addplot[
  scatter,
  only marks,
  point meta=explicit symbolic,
  scatter/classes={
    def={mark=square*,xscale=$xscale,yscale=$yscale,speedupcolor$i}
  }%
]
table[meta=label] {
tw n label
" >> $PLOTFILE

cat $tmpfile >> $PLOTFILE

label="[$begin, $end)"
if [[ $end == $inf ]]; then
    label="\$\ge\$ $begin\,x"
elif [[ $end == 1.05 ]]; then
    label="\$<\$ 5\,\%"
fi
echo "
}; % end of table
\addlegendentry{$label}
" >> $PLOTFILE
fi
rm $tmpfile
done


#=----------------------------------------------------------------------------=#
# Cuckoo tag size
#=----------------------------------------------------------------------------=#
ts=`$SQL "select distinct json_extract(filter_info, '$.tag_bits') as tag_bits from cf_skyline order by json_extract(filter_info, '$.tag_bits');"`

# Create / empty plot file
PLOTFILE="${OUTDIR}/skyline_cf_tag_size.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

plotcolor=0
for t in $ts; do
plotcolor=$((plotcolor + 1))

tmpfile=`mktemp`
echo ".mode tab
select tw, n, 'def' from cf_skyline where json_extract(filter_info, '$.tag_bits') = $t order by tw, n;" | $SQL > $tmpfile
outputsize=`cat $tmpfile | wc -l`
if [[ $outputsize != 0 ]]; then
echo "
% tag size: '$t'
\addplot[
  scatter,
  only marks,
  point meta=explicit symbolic,
  scatter/classes={
    def={mark=square*,xscale=$xscale,yscale=$yscale,blocksizecolor$plotcolor}
  }%
]
table[meta=label] {
tw n label
" >> $PLOTFILE

cat $tmpfile >> $PLOTFILE

echo "
}; % end of table
\addlegendentry{$t Bits\,\,}
" | sed 's/_/ /g' >> $PLOTFILE
fi
rm $tmpfile
done


#=----------------------------------------------------------------------------=#
# Cuckoo associativity
#=----------------------------------------------------------------------------=#
as=`$SQL "select distinct json_extract(filter_info, '$.associativity') as associativity from cf_skyline order by associativity;"`

# Create / empty plot file
PLOTFILE="${OUTDIR}/skyline_cf_associativity.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

plotcolor=0
for a in $as; do
plotcolor=$((plotcolor + 1))

tmpfile=`mktemp`
echo ".mode tab
select tw, n, 'def' from cf_skyline where json_extract(filter_info, '$.associativity') = $a order by tw, n;" | $SQL > $tmpfile
outputsize=`cat $tmpfile | wc -l`
if [[ $outputsize != 0 ]]; then
echo "
% associativity: '$a'
\addplot[
  scatter,
  only marks,
  point meta=explicit symbolic,
  scatter/classes={
    def={mark=square*,xscale=$xscale,yscale=$yscale,blocksizecolor$plotcolor}
  }%
]
table[meta=label] {
tw n label
" >> $PLOTFILE

cat $tmpfile >> $PLOTFILE

echo "
}; % end of table
\addlegendentry{$a \,\,}
" | sed 's/_/ /g' >> $PLOTFILE
fi
rm $tmpfile
done


#=----------------------------------------------------------------------------=#
# Memory footprint
#=----------------------------------------------------------------------------=#

inf=1073741824
#declare -a sizes=(  0       32768     1048576    10485760         67108864         134217728         268435456            $inf)
#declare -a labels=("" "$\le$\,L1" "$\le$\,L2" "$\le$\,L3" "$\le$\,64\,MiB" "$\le$\,128\,MiB" "$\le$\,256\,MiB" "$>$\,256\,MiB")

declare -a sizes
declare -a labels
i=0
sizes[$i]=0
labels[$i]=""
((i++))

cache_levels=`./get_cache_size`
for ((cache_level=1; cache_level<=${cache_levels}; cache_level++)); do
    sizes[$i]=`./get_cache_size $cache_level`
    labels[$i]="$\le$\,L${cache_level}"
    ((i++))
done

sizes[$i]=67108864
labels[$i]="$\le$\,64\,MiB"
((i++))
sizes[$i]=134217728
labels[$i]="$\le$\,128\,MiB"
((i++))
sizes[$i]=268435456
labels[$i]="$\le$\,256\,MiB"
((i++))
sizes[$i]=$inf
labels[$i]="$>$\,256\,MiB"
((i++))

echo "memory footprint granularity:"
echo "   size=${sizes[*]}"
echo " labels=${labels[*]}"

len=${#sizes[@]}

# Bloom filter
PLOTFILE="${OUTDIR}/skyline_memory_footprint.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"


for ((i=1; i<${len}; i++)); do

begin_idx=$((i-1))
end_idx=i
begin=${sizes[$begin_idx]}
end=${sizes[$end_idx]}

tmpfile=`mktemp`
echo ".mode tab
select tw, n, 'def' from bbf_skyline where json_extract(filter_info, '$.size') > $begin and json_extract(filter_info, '$.size') <= $end order by tw, n;" | $SQL >> $tmpfile
outputsize=`cat $tmpfile | wc -l`
if [[ $outputsize != 0 ]]; then
echo "
% size: '[$begin, $end)'
\addplot[
  scatter,
  only marks,
  point meta=explicit symbolic,
  scatter/classes={
    def={mark=square*,xscale=$xscale,yscale=$yscale,memfootprintcolor$i}
  }%
]
table[meta=label] {
tw n label
" >> $PLOTFILE

cat $tmpfile >> $PLOTFILE

label=${labels[$i]}
echo "
}; % end of table
\addlegendentry{$label}
" >> $PLOTFILE

fi
rm $tmpfile
done

# Cuckoo filter
PLOTFILE="${OUTDIR}/skyline_cf_memory_footprint.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

for ((i=1; i<${len}; i++)); do

begin_idx=$((i-1))
end_idx=i
begin=${sizes[$begin_idx]}
end=${sizes[$end_idx]}

tmpfile=`mktemp`
echo ".mode tab
select tw, n, 'def' from cf_skyline where json_extract(filter_info, '$.size') > $begin and json_extract(filter_info, '$.size') <= $end order by tw, n;" | $SQL >> $tmpfile
outputsize=`cat $tmpfile | wc -l`
if [[ $outputsize != 0 ]]; then
echo "
% size: '[$begin, $end)'
\addplot[
  scatter,
  only marks,
  point meta=explicit symbolic,
  scatter/classes={
    def={mark=square*,xscale=$xscale,yscale=$yscale,memfootprintcolor$i}
  }%
]
table[meta=label] {
tw n label
" >> $PLOTFILE

cat $tmpfile >> $PLOTFILE

label=${labels[$i]}
echo "
}; % end of table
\addlegendentry{$label}
" >> $PLOTFILE
fi
rm $tmpfile
done


#=----------------------------------------------------------------------------=#
# Addressing modes
#=----------------------------------------------------------------------------=#
# Create / empty plot file
PLOTFILE="${OUTDIR}/skyline_cf_addr.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

plotcolor=0
for addr in pow2 magic; do
plotcolor=$((plotcolor + 1))

tmpfile=`mktemp`
echo ".mode tab
select tw, n, 'def' from cf_skyline where json_extract(filter_info, '$.addr') = '$addr' order by tw, n;" | $SQL >> $tmpfile
outputsize=`cat $tmpfile | wc -l`
if [[ $outputsize != 0 ]]; then
echo "
% addr: '$addr'
\addplot[
  scatter,
  only marks,
  point meta=explicit symbolic,
  scatter/classes={
    def={mark=square*,xscale=$xscale,yscale=$yscale,addrcolor$plotcolor}
  }%
]
table[meta=label] {
tw n label
" >> $PLOTFILE

cat $tmpfile >> $PLOTFILE


addr_name="n/a"
if [[ $addr == "pow2" ]]; then
    addr_name="Power of two"
elif [[ $addr == "magic" ]]; then
    addr_name="Magic"
fi
echo "
}; % end of table
\addlegendentry{$addr_name}
" >> $PLOTFILE

fi
rm $tmpfile
done


#=----------------------------------------------------------------------------=#
# FPR
#=----------------------------------------------------------------------------=#
echo "
drop table if exists fprranges;
CREATE TABLE fprranges (begin numeric(18,10), end numeric(18,10));
INSERT INTO fprranges VALUES(0.00001,0.0001);
INSERT INTO fprranges VALUES(0.0001,0.001);
INSERT INTO fprranges VALUES(0.001,0.01);
INSERT INTO fprranges VALUES(0.01,0.1);
INSERT INTO fprranges VALUES(0.1,1);

drop table if exists fpr_skyline;
create table fpr_skyline as
select *,
  (select filter from
     (select filter, (r.cycles_per_lookup + r.fpr * tw_values.tw) as overhead from results r where r.n = n_values.n
      order by overhead limit 1)
  ) as filter,
  (select fpr from
     (select fpr, (r.cycles_per_lookup + r.fpr * tw_values.tw) as overhead from results r where r.n = n_values.n
      order by overhead limit 1)
  ) as fpr
from n_values, tw_values, fprranges
where fpr >= fprranges.begin and fpr < fprranges.end
;
" | $SQL

# Create / empty plot file
PLOTFILE="${OUTDIR}/skyline_fpr.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

#r_s=`$SQL "select distinct begin from fpr_skyline order by begin;"`
r_s="0.00001
0.0001
0.001
0.01
0.1"


plotcolor=6
for r in $r_s; do
plotcolor=$((plotcolor - 1))
echo "
% fpr range: '$r'
\addplot[
  scatter,
  only marks,
  point meta=explicit symbolic,
  scatter/classes={
    def={mark=square*,xscale=$xscale,yscale=$yscale,svzcolor$plotcolor}
  }%
]
table[meta=label] {
tw n label
" >> $PLOTFILE

echo ".mode tab
select tw, n, 'def' from fpr_skyline where begin = $r and n < 198216249 and tw < 2147483648 order by tw, n;" | $SQL >> $PLOTFILE

r_name=`$SQL "select '[' || '$r' || ', ' || end || ')' from fprranges where begin = $r;"`

echo "
}; % end of table
\addlegendentry{$r_name}
" >> $PLOTFILE

done


#rm $DBFILE
echo "Database file: $DBFILE"


exit 0

# ----------------------------------
# Memory footprint for selected n's (DEPRECATED)
# ----------------------------------

PLOTFILE="${OUTDIR}/skyline_memory_footprint_detail.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

#declare -a sizes=(       115097       920781     10417458)
#declare -a labels=("\$n=10^5\$" "\$n=10^6\$" "\$n=10^7\$")

declare -a sizes=(       115097       920781     10417458    103496016)
declare -a labels=("\$n=10^5\$" "\$n=10^6\$" "\$n=10^7\$" "\$n=10^8\$")

#declare -a sizes=(        12098        96785       920781     12388515)
#declare -a labels=("\$n=10^4\$" "\$n=10^5\$" "\$n=10^6\$" "\$n=10^7\$")


#declare -a sizes=(         9741       115097       920781     10417458    103496016)
#declare -a labels=("\$n=10^4\$" "\$n=10^5\$" "\$n=10^6\$" "\$n=10^7\$" "\$n=10^8\$")

len=${#sizes[@]}

for ((i=0; i<${len}; i++)); do

n=${sizes[$i]}

color_idx=$((i+1))
echo "
% n: '$n'
\addplot[
  draw=plotcolor$color_idx
]
table  {
tw size
" >> $PLOTFILE

echo ".mode tab
select tw, json_extract(filter_info, '$.size')/1024.0/1024.0 from bbf_skyline where n = $n order by tw;" | $SQL >> $PLOTFILE

label=${labels[$i]}
echo "
}; % end of table
\addlegendentry{$label}
" >> $PLOTFILE

done
# -------------------------------
PLOTFILE="${OUTDIR}/skyline_cf_memory_footprint_detail.tex"
echo -n "" > $PLOTFILE
echo "writing plot file: `basename $PLOTFILE`"

#declare -a sizes=(       115097       920781     10417458)
#declare -a labels=("\$n=10^5\$" "\$n=10^6\$" "\$n=10^7\$")
declare -a sizes=(       115097       920781     10417458    103496016)
declare -a labels=("\$n=10^5\$" "\$n=10^6\$" "\$n=10^7\$" "\$n=10^8\$")
#declare -a sizes=(         9741       115097       920781     10417458    103496016)
#declare -a labels=("\$n=10^4\$" "\$n=10^5\$" "\$n=10^6\$" "\$n=10^7\$" "\$n=10^8\$")

len=${#sizes[@]}

for ((i=0; i<${len}; i++)); do

n=${sizes[$i]}

color_idx=$((i+1))
echo "
% n: '$n'
\addplot[
  draw=plotcolor$color_idx
]
table  {
tw size
" >> $PLOTFILE

echo ".mode tab
select tw, json_extract(filter_info, '$.size')/1024.0/1024.0 from cf_skyline where n = $n order by tw;" | $SQL >> $PLOTFILE

label=${labels[$i]}
echo "
}; % end of table
\addlegendentry{$label}
" >> $PLOTFILE

done
