#!/bin/bash

# Download and crop ISIMIP3b bias-adjusted daily climate data.
# Updated for the 5 additional models and scenarios: historical, ssp126, ssp245, ssp370, ssp585.

set -u

# Optional default environment activation. Comment out if not needed.
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate py_data_analysis 2>/dev/null || true
fi

usage() {
    echo "Usage: $0 -m model_choices -v variable -s scenario [-x \"xmin xmax\"] [-y \"ymin ymax\"]"
    echo
    echo "Download ISIMIP3b climate data with specified parameters."
    echo
    echo "Options:"
    echo "  -m   model_choices    Model choice(s), e.g. all, MIROC6, EC-Earth3, CanESM5, CESM2-WACCM, IITM-ESM, or a quoted list."
    echo "                         Available: MIROC6, EC-Earth3, CanESM5, CESM2-WACCM, IITM-ESM, all"
    echo "                         Aliases accepted: miroc6, ec-earth, canesm, cesm2, iitm"
    echo "  -v   variable          One or more variables separated by spaces: hurs, huss, pr, prsn, ps, tas, tasmax, tasmin"
    echo "                         Example: -v \"pr tas tasmax tasmin\""
    echo "  -s   scenario          Scenario choice(s): historical, ssp126/126, ssp245/245, ssp370/370, ssp585/585, all"
    echo "                         Example: -s all"
    echo "  -x   \"xmin xmax\"     Longitude bounds for cropping. Default: \"-180 180\""
    echo "  -y   \"ymin ymax\"     Latitude bounds for cropping. Default: \"-90 90\""
    echo "  -h                     Show this help message"
    exit 1
}

# Model set requested: only the five additional models not already downloaded.
declare -a all_models=(
    "MIROC6"
    "EC-Earth3"
    "CanESM5"
    "CESM2-WACCM"
    "IITM-ESM"
)

declare -a allowed_variables=("hurs" "huss" "pr" "prsn" "ps" "tas" "tasmax" "tasmin")
declare -a all_scenarios=("historical" "ssp126" "ssp245" "ssp370" "ssp585")

contains() {
    local item=$1
    shift
    local x
    for x in "$@"; do
        [[ "$x" == "$item" ]] && return 0
    done
    return 1
}

normalize_model() {
    # Accept a few short/lowercase aliases but return the official ISIMIP folder name.
    case "$1" in
        MIROC6|miroc6) echo "MIROC6" ;;
        EC-Earth3|ec-earth3|EC-Earth|ec-earth) echo "EC-Earth3" ;;
        CanESM5|canesm5|CanESM|canesm) echo "CanESM5" ;;
        CESM2-WACCM|cesm2-waccm|CESM2|cesm2) echo "CESM2-WACCM" ;;
        IITM-ESM|iitm-esm|IITM|iitm) echo "IITM-ESM" ;;
        all) echo "all" ;;
        *) echo "$1" ;;
    esac
}

normalize_scenario() {
    case "$1" in
        historical) echo "historical" ;;
        126|ssp126) echo "ssp126" ;;
        245|ssp245) echo "ssp245" ;;
        370|ssp370) echo "ssp370" ;;
        585|ssp585) echo "ssp585" ;;
        all) echo "all" ;;
        *) echo "$1" ;;
    esac
}

validate_model_choices() {
    local input=$1
    local models=()
    IFS=' ' read -r -a models <<< "$input"

    for model_raw in "${models[@]}"; do
        local model
        model=$(normalize_model "$model_raw")
        if [[ "$model" == "all" ]]; then
            continue
        fi
        if ! contains "$model" "${all_models[@]}"; then
            echo "Error: Invalid model choice: $model_raw"
            echo "Allowed models: ${all_models[*]}, all"
            usage
        fi
    done
}

validate_variable() {
    local input=$1
    local variables=()
    IFS=' ' read -r -a variables <<< "$input"

    for variable in "${variables[@]}"; do
        if ! contains "$variable" "${allowed_variables[@]}"; then
            echo "Error: Invalid variable: $variable"
            echo "Allowed variables: ${allowed_variables[*]}"
            usage
        fi
    done
}

validate_scenario() {
    local input=$1
    local scenarios=()
    IFS=' ' read -r -a scenarios <<< "$input"

    for scenario_raw in "${scenarios[@]}"; do
        local scenario
        scenario=$(normalize_scenario "$scenario_raw")
        if [[ "$scenario" == "all" ]]; then
            continue
        fi
        if ! contains "$scenario" "${all_scenarios[@]}"; then
            echo "Error: Invalid scenario: $scenario_raw"
            echo "Allowed scenarios: historical, ssp126/126, ssp245/245, ssp370/370, ssp585/585, all"
            usage
        fi
    done
}

# Defaults for cropping
xlim=(-180 180)
ylim=(-90 90)
model_choices=""
variable=""
scenario=""

while getopts ":h:m:v:s:x:y:" opt; do
    case ${opt} in
        h)
            usage
            ;;
        m)
            validate_model_choices "$OPTARG"
            model_choices=$OPTARG
            ;;
        v)
            validate_variable "$OPTARG"
            variable=$OPTARG
            ;;
        s)
            validate_scenario "$OPTARG"
            scenario=$OPTARG
            ;;
        x)
            xlim=($OPTARG)
            ;;
        y)
            ylim=($OPTARG)
            ;;
        \?)
            echo "Invalid option: -$OPTARG" 1>&2
            exit 1
            ;;
        :)
            echo "Invalid option: -$OPTARG requires an argument" 1>&2
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

if [[ -z "$model_choices" || -z "$variable" || -z "$scenario" ]]; then
    echo "Error: Missing arguments. Options -m, -v and -s are required."
    usage
fi

if [[ ${#xlim[@]} -ne 2 || ${#ylim[@]} -ne 2 ]]; then
    echo "Error: -x and -y must each contain two values, e.g. -x \"2 7\" -y \"49 52\""
    exit 1
fi

for cmd in wget cdo grep cut head find; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "$cmd could not be found, please install or load it first."
        exit 1
    fi
done

if ! command -v parallel &> /dev/null; then
    echo "GNU parallel could not be found, please install or load it first."
    exit 1
fi

# Ask user for conda environment for NCML generation only.
echo "Please specify a conda environment to activate if ncml files are to be generated."
echo "Type 'no' if you only want to download and crop the data."
read -r -p "Conda environment: " conda_env

if [[ "$conda_env" == "no" ]]; then
    echo "Skipping conda environment activation and ncml creation."
else
    echo "Activating conda environment $conda_env..."
    source activate "$conda_env"
    if [[ $? -ne 0 ]]; then
        echo "Failed to activate conda environment $conda_env. Exiting."
        exit 1
    fi
    echo "Successfully activated conda environment $conda_env."
fi

resolve_models() {
    local input=$1
    local models=()
    IFS=' ' read -r -a models <<< "$input"

    if contains "all" "${models[@]}"; then
        printf '%s\n' "${all_models[@]}"
    else
        printf '%s\n' "${models[@]}"
    fi
}

resolve_scenarios() {
    local input=$1
    local scenarios=()
    IFS=' ' read -r -a scenarios <<< "$input"

    if contains "all" "${scenarios[@]}"; then
        printf '%s\n' "${all_scenarios[@]}"
    else
        local s
        for s in "${scenarios[@]}"; do
            normalize_scenario "$s"
        done
    fi
}

# Function to download files and crop them.
download_model_files() {
    local model=$1
    local variable_string=$2
    local scenario_string=$3
    local xmin=$4
    local xmax=$5
    local ymin=$6
    local ymax=$7

    local variables=()
    local scenarios=()
    IFS=' ' read -r -a variables <<< "$variable_string"
    mapfile -t scenarios < <(resolve_scenarios "$scenario_string")

    # User-provided base URL. Directory structure is model/scenario/file.nc.
    local base_url="https://files.isimip.org/ISIMIP3b/SecondaryInputData/climate/atmosphere/bias-adjusted/global/daily"
    local lower_model
    lower_model=$(echo "$model" | tr '[:upper:]' '[:lower:]')

    for variable in "${variables[@]}"; do
        for scenario in "${scenarios[@]}"; do
            mkdir -p "${model}/${scenario}"

            local starts=()
            if [[ "$scenario" == "historical" ]]; then
                starts=(1971 1981 1991 2001 2011)
            else
                starts=(2021 2031 2041 2051 2061 2071 2081 2091)
            fi

            local remote_dir="${base_url}/${model}/${scenario}/"
            echo "Checking ${model} / ${scenario} / ${variable}"

            local year
            for year in "${starts[@]}"; do
                local end_year=$((year + 9))
                if [[ "$scenario" == "historical" && "$end_year" -gt 2014 ]]; then
                    end_year=2014
                fi

                local regex="^${lower_model}_.+_w5e5_${scenario}_${variable}_global_daily_${year}_${end_year}\.nc$"
                local local_pattern="${lower_model}_*_w5e5_${scenario}_${variable}_global_daily_${year}_${end_year}.nc"
                local file
                file=$(find "${model}/${scenario}" -maxdepth 1 -name "$local_pattern" ! -name "*_cropped.nc" | head -n 1 || true)

                if [[ -z "$file" ]]; then
                    echo "Searching remote directory: $remote_dir"
                    local fname
                    fname=$(wget -qO- "$remote_dir" | grep -oE 'href="[^"]+\.nc"' | cut -d'"' -f2 | grep -E "$regex" | head -n 1 || true)

                    if [[ -z "$fname" ]]; then
                        echo "No matching file found for ${model} ${scenario} ${variable} ${year}-${end_year}. Skipping."
                        continue
                    fi

                    file="${model}/${scenario}/${fname}"
                    if [[ ! -f "$file" ]]; then
                        echo "Downloading ${remote_dir}${fname}"
                        wget -c "${remote_dir}${fname}" -O "$file" || {
                            echo "Failed to download ${remote_dir}${fname}"
                            rm -f "$file"
                            continue
                        }
                    fi
                else
                    echo "File $file already exists, skipping download."
                fi

                local output_file="${file%.nc}_cropped.nc"
                if [[ -f "$output_file" ]]; then
                    echo "Cropped file $output_file already exists, skipping crop."
                    rm -f "$file"
                    continue
                fi

                echo "Cropping: $file"
                if cdo -O sellonlatbox,"${xmin}","${xmax}","${ymin}","${ymax}" "$file" "$output_file"; then
                    echo "Cropping successful: $output_file"
                    rm -f "$file"
                else
                    echo "Error cropping file: $file"
                fi
            done
        done
    done
}

export -f download_model_files
export -f resolve_scenarios
export -f normalize_scenario
export -f normalize_model

mapfile -t selected_models < <(resolve_models "$model_choices")
mapfile -t selected_scenarios < <(resolve_scenarios "$scenario")

# Debug statements
echo "Models: ${selected_models[*]}"
echo "Variables: $variable"
echo "Scenarios: ${selected_scenarios[*]}"
echo "Longitude bounds: ${xlim[*]}"
echo "Latitude bounds: ${ylim[*]}"

# Download and crop files. Adjust --jobs depending on available bandwidth and storage.
echo "Downloading and cropping files. This can take a while..."
parallel --jobs 5 --delay 1 \
    "download_model_files {} '$variable' '$scenario' '${xlim[0]}' '${xlim[1]}' '${ylim[0]}' '${ylim[1]}'" \
    ::: "${selected_models[@]}"

create_ncml() {
    local model_input=$1
    local scenario_input=$2
    local models=()
    local scenarios=()

    mapfile -t models < <(resolve_models "$model_input")
    mapfile -t scenarios < <(resolve_scenarios "$scenario_input")

    for model in "${models[@]}"; do
        for scenario in "${scenarios[@]}"; do
            if [[ -d "${model}/${scenario}" ]]; then
                mkdir -p "ncml/${scenario}"
                Rscript -e "library(loadeR); makeAggregatedDataset(source.dir=paste0('./', '${model}', '/', '${scenario}'), ncml.file=paste0('./ncml/', '${scenario}', '/', '${model}', '_', '${scenario}', '.ncml'))"
            else
                echo "Directory ${model}/${scenario} does not exist."
            fi
        done
    done
}

export -f create_ncml
export -f resolve_models

if [[ "$conda_env" == "no" ]]; then
    echo "Skipping ncml file creation."
else
    create_ncml "$model_choices" "$scenario"
fi
