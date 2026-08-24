#!/bin/bash
# Query S2 for refs, compact output
declare -A refs
refs[1]="DOI:10.1006/nimg.2002.1087"
refs[2]="DOI:10.1146/annurev-neuro-071013-014030"
refs[3]="DOI:10.1016/j.compbiomed.2011.06.020"
refs[4]="DOI:10.1016/j.neuroimage.2013.11.001"
refs[5]="DOI:10.1109/EMBC58623.2025.11253529"
refs[6]="DOI:10.1109/TAFFC.2018.2824988"
refs[7]="DOI:10.1109/TNSRE.2025.3603190"
refs[8]="DOI:10.1016/j.bspc.2025.107536"
refs[9]="search?query=DDSPR+dynamic+domain+selection+pseudo-label+refinement+cross-subject+EEG+emotion&limit=3"
refs[10]="DOI:10.1007/978-981-97-8499-8_28"
refs[11]="DOI:10.1109/TKDE.2022.3178128"
refs[12]="arXiv:1505.07818"
refs[13]="DOI:10.1007/978-3-319-46493-0_27"
refs[14]="DOI:10.1088/1741-2552/aace8c"
refs[15]="DOI:10.1109/TAFFC.2022.3220943"
refs[16]="DOI:10.1109/TAFFC.2018.2825452"
refs[17]="DOI:10.1109/TNSRE.2022.3230250"
refs[18]="DOI:10.1016/j.neuroimage.2023.120209"
refs[19]="DOI:10.1609/aaai.v35i9.16936"
refs[20]="arXiv:1909.01377"
refs[21]="arXiv:2006.11959"
refs[22]="DOI:10.1609/aaai.v36i6.20614"
refs[23]="arXiv:1612.00410"
refs[24]="arXiv:1503.02531"
refs[25]="DOI:10.1038/4580"
refs[26]="DOI:10.1007/s10994-009-5152-4"
refs[27]="search?query=Measuring+phase+synchrony+in+brain+signals&limit=3"
refs[28]="search?query=Sur+les+operations+dans+les+ensembles+abstraits+et+leur+application+aux+equations+integrales&limit=3"
refs[29]="DOI:10.1023/A:1024068626366"
refs[30]="search?query=The+Design+of+Experiments+Fisher&limit=3"
refs[31]="search?query=A+simple+sequentially+rejective+multiple+test+procedure&limit=3"
refs[32]="search?query=Statistical+Methods+Snedecor+Cochran&limit=3"
refs[33]="DOI:10.1088/1741-2560/12/3/031001"
for n in $(seq 1 33); do
  q="${refs[$n]}"
  echo "===== REF $n : $q"
  curl -s --max-time 30 "https://api.semanticscholar.org/graph/v1/paper/${q}&fields=title,authors,year,venue,externalIds,openAccessPdf,isOpenAccess" | head -c 1200
  echo
  sleep 1
done
