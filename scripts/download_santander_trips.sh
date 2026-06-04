#!/bin/bash
# Download Santander Cycles trip data: 86 files (350-434) covering Jan 2023 - Dec 2025
BASE="https://cycling.data.tfl.gov.uk/usage-stats"
DIR="data/raw/london/transport/santander_cycles/trips"
mkdir -p "$DIR"

FILES=(
"350JourneyDataExtract26Dec2022-01Jan2023.csv"
"351JourneyDataExtract02Jan2023-08Jan2023.csv"
"352JourneyDataExtract09Jan2023-15Jan2023.csv"
"353JourneyDataExtract16Jan2023-22Jan2023.csv"
"354JourneyDataExtract23Jan2023-29Jan2023.csv"
"355JourneyDataExtract30Jan2023-05Feb2023.csv"
"356JourneyDataExtract06Feb2023-12Feb2023.csv"
"357JourneyDataExtract13Feb2023-19Feb2023.csv"
"358JourneyDataExtract20Feb2023-26Feb2023.csv"
"359JourneyDataExtract27Feb2023-05Mar2023.csv"
"360JourneyDataExtract06Mar2023-12Mar2023.csv"
"361JourneyDataExtract13Mar2023-19Mar2023.csv"
"362JourneyDataExtract20Mar2023-26Mar2023.csv"
"363JourneyDataExtract27Mar2023-02Apr2023.csv"
"364JourneyDataExtract03Apr2023-09Apr2023.csv"
"365JourneyDataExtract10Apr2023-16Apr2023.csv"
"366JourneyDataExtract17Apr2023-23Apr2023.csv"
"367JourneyDataExtract24Apr2023-30Apr2023.csv"
"368JourneyDataExtract01May2023-07May2023.csv"
"369JourneyDataExtract08May2023-14May2023.csv"
"370JourneyDataExtract15May2023-21May2023.csv"
"371JourneyDataExtract22May2023-28May2023.csv"
"372JourneyDataExtract29May2023-04Jun2023.csv"
"373JourneyDataExtract05Jun2023-11Jun2023.csv"
"374JourneyDataExtract12Jun2023-18Jun2023.csv"
"375JourneyDataExtract19Jun2023-30Jun2023.csv"
"376JourneyDataExtract01Jul2023-14Jul2023.csv"
"377JourneyDataExtract15Jul2023-31Jul2023.csv"
"378JourneyDataExtract01Aug2023-14Aug2023.csv"
"378JourneyDataExtract15Aug2023-31Aug2023.csv"
"379JourneyDataExtract01Sep2023-14Sep2023.csv"
"380JourneyDataExtract15Sep2023-30Sep2023.csv"
"381JourneyDataExtract01Oct2023-14Oct2023.csv"
"382JourneyDataExtract15Oct2023-31Oct2023.csv"
"383JourneyDataExtract01Nov2023-14Nov2023.csv"
"384JourneyDataExtract15Nov2023-30Nov2023.csv"
"385JourneyDataExtract01Dec2023-14Dec2023.csv"
"386JourneyDataExtract15Dec2023-31Dec2023.csv"
"387JourneyDataExtract01Jan2024-14Jan2024.csv"
"388JourneyDataExtract15Jan2024-31Jan2024.csv"
"389JourneyDataExtract01Feb2024-14Feb2024.csv"
"390JourneyDataExtract15Feb2024-29Feb2024.csv"
"391JourneyDataExtract01Mar2024-14Mar2024.csv"
"392JourneyDataExtract15Mar2024-31Mar2024.csv"
"393JourneyDataExtract01Apr2024-14Apr2024.csv"
"394JourneyDataExtract15Apr2024-30Apr2024.csv"
"395JourneyDataExtract01May2024-14May2024.csv"
"396JourneyDataExtract15May2024-31May2024.csv"
"397JourneyDataExtract01Jun2024-14Jun2024.csv"
"398JourneyDataExtract15Jun2024-30Jun2024.csv"
"399JourneyDataExtract01Jul2024-14Jul2024.csv"
"400JourneyDataExtract15Jul2024-31Jul2024.csv"
"401JourneyDataExtract01Aug2024-14Aug2024.csv"
"402JourneyDataExtract15Aug2024-26Aug2024.csv"
"403JourneyDataExtract27Aug2024-17Sep2024.csv"
"404JourneyDataExtract18Sep2024-30Sep2024.csv"
"405JourneyDataExtract01Oct2024-14Oct2024.csv"
"406JourneyDataExtract15Oct2024-31Oct2024.csv"
"407JourneyDataExtract01Nov2024-14Nov2024.csv"
"408JourneyDataExtract15Nov2024-30Nov2024.csv"
"409JourneyDataExtract01Dec2024-14Dec2024.csv"
"410JourneyDataExtract15Dec2024-31Dec2024.csv"
"411JourneyDataExtract01Jan2025-14Jan2025.csv"
"412JourneyDataExtract15Jan2025-31Jan2025.csv"
"413JourneyDataExtract01Feb2025-14Feb2025.csv"
"414JourneyDataExtract15Feb2025-28Feb2025.csv"
"415JourneyDataExtract01Mar2025-14Mar2025.csv"
"416JourneyDataExtract15Mar2025-31Mar2025.csv"
"417JourneyDataExtract01Apr2025-14Apr2025.csv"
"418JourneyDataExtract15Apr2025-30Apr2025.csv"
"419JourneyDataExtract01May2025-14May2025.csv"
"420JourneyDataExtract14May2025-31May2025.csv"
"421JourneyDataExtract01Jun2025-15Jun2025.csv"
"422JourneyDataExtract15Jun2025-30Jun2025.csv"
"423JourneyDataExtract01Jul2025-15Jul2025.csv"
"424JourneyDataExtract16Jul2025-31Jul2025.csv"
"425JourneyDataExtract01Aug2025-15Aug2025.csv"
"426JourneyDataExtract16Aug2025-31Aug2025.csv"
"427JourneyDataExtract01Sep2025-15Sep2025.csv"
"428JourneyDataExtract16Sep2025-30Sep2025.csv"
"429JourneyDataExtract01Oct2025-15Oct2025.csv"
"430JourneyDataExtract16Oct2025-31Oct2025.csv"
"431JourneyDataExtract01Nov2025-15Nov2025.csv"
"432JourneyDataExtract16Nov2025-30Nov2025.csv"
"433JourneyDataExtract01Dec2025-15Dec2025.csv"
"434JourneyDataExtract16Dec2025-31Dec2025.csv"
)

total=${#FILES[@]}
success=0
fail=0
skip=0

echo "Downloading $total Santander Cycles trip files..."
echo ""

for f in "${FILES[@]}"; do
    outpath="$DIR/$f"
    if [ -f "$outpath" ] && [ -s "$outpath" ]; then
        ((skip++))
        continue
    fi
    
    url="$BASE/$f"
    code=$(curl -sL -o "$outpath" -w "%{http_code}" "$url")
    
    if [ "$code" = "200" ] && [ -s "$outpath" ]; then
        ((success++))
        echo "  [OK] $f"
    else
        ((fail++))
        echo "  [FAIL:$code] $f"
        rm -f "$outpath"
    fi
done

echo ""
echo "Done: $success downloaded, $skip skipped, $fail failed (of $total total)"
echo ""
echo "Total disk usage:"
du -sh "$DIR"
