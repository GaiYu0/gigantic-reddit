# TODO: wget failure
from=$((12 * $1 + $2))
to=$((12 * $3 + $4))
for i in $(seq $from $to); do
    year=$((i / 12))
    month=$((i % 12))
    if [ $month -eq 0 ]; then
        year=$(($year - 1))
        month=12
    fi
    fn=RC_${year}-${month}
    if [ $i -lt $((12 * 2017 + 12)) ]; then
        wget $5 https://files.pushshift.io/reddit/comments/${fn}.bz2
        bzip2 -d ${fn}.bz2
    else
        wget $5 https://files.pushshift.io/reddit/comments/${fn}.xz
        unxz ${fn}.xz
    fi
    if [ $i -eq $from ]; then
        fn_list=$fn
    else
        fn_list="${fn_list} ${fn}"
    fi
done
cat $fn_list > RC_${from}-${to}
