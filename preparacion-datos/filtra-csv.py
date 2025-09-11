#!/usr/bin/env python3


import os
import sys
import pandas as pd


def pedir_archivo() -> str:
    nombre = input("Nombre del archivo CSV: ").strip()
    if not nombre.lower().endswith(".csv"):
        nombre += ".csv"
    if not os.path.isfile(nombre):
        sys.exit(f"Archivo '{nombre}' no encontrado.")
    return nombre


def elegir_columna(df: pd.DataFrame) -> str:
    for idx, col in enumerate(df.columns, 1):
        print(f"{idx:>2}. {col}")
    while True:
        sel = input("Seleccione columna (numero): ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(df.columns):
            return df.columns[int(sel) - 1]
        print("Opcion no valida; intente de nuevo.")


def elegir_valor(valores) -> str:
    for idx, val in enumerate(valores, 1):
        print(f"{idx:>2}. {val}")
    sel = input("Valor a eliminar (numero o texto): ").strip()
    if sel.isdigit():
        pos = int(sel) - 1
        if 0 <= pos < len(valores):
            return valores[pos]
    return sel


def main() -> None:
    csv_nombre = pedir_archivo()
    df = pd.read_csv(csv_nombre)

    columna = elegir_columna(df)
    valores_unicos = df[columna].dropna().unique()
    valor_a_eliminar = elegir_valor(valores_unicos)

    filtrado = df[df[columna] != valor_a_eliminar]

    base, ext = os.path.splitext(csv_nombre)
    salida = f"{base}-tr{ext}"
    filtrado.to_csv(salida, index=False)

    eliminadas = len(df) - len(filtrado)
    print(f"{eliminadas} filas eliminadas.")
    print(f"Resultado guardado en '{salida}'.")


if __name__ == "__main__":
    main()
