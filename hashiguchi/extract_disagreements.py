#!/usr/bin/env python3
"""
CSVファイルを統合して、予測が分かれている行を抽出し、多数決で最終予測を作成するスクリプト

このスクリプトは以下の処理を行います：
1. 指定されたCSVファイル（PassengerId, Transportedの形式）を統合
2. 予測が分かれている行を抽出してCSVファイルを作成
3. 奇数個のファイルがある場合、多数決でfinal_submission.csvを作成
"""

import pandas as pd
import sys
import os
import glob
from pathlib import Path

# ===== ここで設定を指定 =====
CSV_FILES = [
    "submission_catboost.csv",
    "submission_lightgbm.csv", 
    "submission_catboost_many_features.csv",
    "submission_10_19.csv",
    "submission_80.csv",
]

# 最終的なsubmissionファイルを作成するかどうか
# True: 多数決でfinal_submission.csvも作成
# False: 比較ファイル（disagreements）のみ作成
CREATE_FINAL_SUBMISSION = False
# ==============================

def create_ensemble_prediction(csv_files, final_output_file="final_submission.csv"):
    """
    複数のCSVファイルから多数決で最終予測を作成する
    
    Args:
        csv_files (list): CSVファイルのパスのリスト
        final_output_file (str): 最終予測の出力ファイル名
    
    Returns:
        pd.DataFrame: 最終予測のデータフレーム
    """
    
    if len(csv_files) % 2 == 0:
        print(f"警告: ファイル数が偶数（{len(csv_files)}個）のため、多数決での最終予測は作成しません")
        return None
    
    # 最初のCSVファイルを読み込み
    final_df = pd.read_csv(csv_files[0])
    final_df = final_df.set_index('PassengerId')
    final_df = final_df.rename(columns={'Transported': f'Transported_{Path(csv_files[0]).stem}'})
    
    # 残りのCSVファイルを順次マージ
    for file in csv_files[1:]:
        df = pd.read_csv(file)
        df = df.set_index('PassengerId')
        df = df.rename(columns={'Transported': f'Transported_{Path(file).stem}'})
        final_df = final_df.join(df, how='inner')
    
    # 各行で多数決を実行
    transported_cols = [col for col in final_df.columns if col.startswith('Transported_')]
    
    # 各行でTrueの数を数える
    true_counts = final_df[transported_cols].sum(axis=1)
    # 多数決でTransportedを決定
    final_df['Transported'] = true_counts > (len(transported_cols) / 2)
    
    # 最終的な予測結果を作成
    result_df = final_df[['Transported']].reset_index()
    
    # ファイルに保存
    result_df.to_csv(final_output_file, index=False)
    
    print(f"\n=== 多数決による最終予測 ===")
    print(f"使用したファイル数: {len(csv_files)}")
    print(f"Trueの予測: {result_df['Transported'].sum()}")
    print(f"Falseの予測: {(~result_df['Transported']).sum()}")
    print(f"最終予測を '{final_output_file}' に保存しました")
    
    return result_df

def extract_disagreements(csv_files, output_file="disagreements.csv"):
    """
    複数のCSVファイルから予測が分かれている行を抽出する
    
    Args:
        csv_files (list): CSVファイルのパスのリスト
        output_file (str): 出力ファイル名
    
    Returns:
        pd.DataFrame: 予測が分かれている行のデータフレーム
    """
    
    # ファイルの存在確認
    for file in csv_files:
        if not os.path.exists(file):
            print(f"エラー: ファイル '{file}' が見つかりません")
            return None
    
    # 最初のCSVファイルを読み込み
    merged_df = pd.read_csv(csv_files[0])
    print(f"ファイル '{csv_files[0]}' を読み込みました ({len(merged_df)} 行)")
    
    # PassengerIdをインデックスに設定
    merged_df = merged_df.set_index('PassengerId')
    merged_df = merged_df.rename(columns={'Transported': f'Transported_{Path(csv_files[0]).stem}'})
    
    # 残りのCSVファイルを順次マージ
    for i, file in enumerate(csv_files[1:], 1):
        df = pd.read_csv(file)
        print(f"ファイル '{file}' を読み込みました ({len(df)} 行)")
        
        df = df.set_index('PassengerId')
        df = df.rename(columns={'Transported': f'Transported_{Path(file).stem}'})
        
        # マージ
        merged_df = merged_df.join(df, how='inner')
    
    print(f"統合後のデータ: {len(merged_df)} 行")
    
    # 予測が分かれている行を特定
    # 各行で全ての予測値が同じかどうかをチェック
    transported_cols = [col for col in merged_df.columns if col.startswith('Transported_')]
    
    # 各行で予測が一致しているかをチェック
    # nunique() > 1 の場合、その行では予測が分かれている
    disagreement_mask = merged_df[transported_cols].nunique(axis=1) > 1
    
    disagreements_df = merged_df[disagreement_mask].copy()
    
    print(f"予測が分かれている行: {len(disagreements_df)} 行")
    
    if len(disagreements_df) > 0:
        # 結果を保存
        disagreements_df.reset_index().to_csv(output_file, index=False)
        print(f"結果を '{output_file}' に保存しました")
        
        # 統計情報を表示
        print("\n=== 統計情報 ===")
        print(f"総行数: {len(merged_df)}")
        print(f"予測が分かれている行数: {len(disagreements_df)}")
        print(f"一致率: {(len(merged_df) - len(disagreements_df)) / len(merged_df) * 100:.2f}%")
        
        # 各ファイルの予測分布を表示
        print("\n=== 各ファイルの予測分布（分かれている行のみ） ===")
        for col in transported_cols:
            true_count = (disagreements_df[col] == True).sum()
            false_count = (disagreements_df[col] == False).sum()
            print(f"{col}: True={true_count}, False={false_count}")
            
        # サンプルを表示
        print(f"\n=== サンプル（最初の10行） ===")
        print(disagreements_df.head(10))
        
    else:
        print("予測が分かれている行はありませんでした（全て一致）")
    
    return disagreements_df

def main():
    # スクリプト上部で指定されたCSVファイルを使用
    csv_files = CSV_FILES
    
    # ファイルの存在確認
    existing_files = []
    missing_files = []
    
    for file in csv_files:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            missing_files.append(file)
    
    if missing_files:
        print(f"警告: 以下のファイルが見つかりませんでした:")
        for f in missing_files:
            print(f"  - {f}")
    
    if not existing_files:
        print("エラー: 処理可能なCSVファイルがありません")
        sys.exit(1)
    
    if len(existing_files) != len(csv_files):
        print(f"存在するファイルのみ処理します: {existing_files}")
    
    csv_files = existing_files
    
    # 出力ファイル名を生成（入力ファイル名から）
    file_stems = [Path(f).stem for f in csv_files]
    output_file = f"disagreements_{'_vs_'.join(file_stems)}.csv"
    
    print(f"\n=== CSV統合・予測分析スクリプト ===")
    print(f"入力ファイル: {csv_files}")
    print(f"出力ファイル: {output_file}")
    print()
    
    # 予測が分かれている行を抽出
    disagreements_df = extract_disagreements(csv_files, output_file)
    
    # 設定に応じて最終予測を作成
    if CREATE_FINAL_SUBMISSION:
        if len(csv_files) % 2 == 1:
            final_df = create_ensemble_prediction(csv_files, "final_submission.csv")
        else:
            print(f"\n注意: ファイル数が偶数（{len(csv_files)}個）のため、多数決による最終予測は作成されませんでした")
    else:
        print(f"\n設定により、比較ファイルのみ作成しました（CREATE_FINAL_SUBMISSION = False）")
    
    if disagreements_df is not None and len(disagreements_df) > 0:
        print(f"\n処理完了！予測が分かれている {len(disagreements_df)} 行を '{output_file}' に保存しました。")
    else:
        print("\n処理完了！")

if __name__ == "__main__":
    main()