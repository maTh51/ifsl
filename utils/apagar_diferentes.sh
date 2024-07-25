#!/bin/bash

# Caminho para o diretório contendo os arquivos .txt
DIR="/scratch/matheuspimenta/chesapeake_400/pools"

# Função para verificar a existência dos arquivos listados em um arquivo .txt
check_files() {
    local txt_file="$1"
    local temp_file="${txt_file}.tmp"

    # Ler cada linha do arquivo .txt e verificar se o arquivo existe
    while IFS= read -r line; do
        if [ -f "$line" ]; then
            echo "$line" >> "$temp_file"
        else
            echo "Arquivo não encontrado: $line"
        fi
    done < "$txt_file"

    # Substituir o arquivo original pelo temporário
    mv "$temp_file" "$txt_file"
}

# Iterar sobre cada arquivo .txt no diretório
for txt_file in "$DIR"/*.txt; do
    [ -e "$txt_file" ] || continue
    check_files "$txt_file"
done
