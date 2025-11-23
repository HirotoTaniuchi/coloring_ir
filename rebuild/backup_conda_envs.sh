#!/bin/bash
# ==============================================
# backup_conda_envs.sh
# 全てのconda環境をYAMLファイルとしてバックアップするスクリプト
# 保存先: ./conda_env_backups/
# ==============================================

set -e  # エラーが出たら即停止

# 出力ディレクトリ作成
BACKUP_DIR="/home/usrs/taniuchi/workspace/projects/coloring_ir/rebuild"
mkdir -p "$BACKUP_DIR"

echo "📦 Conda環境一覧を取得中..."
ENVS=$(conda env list | awk '{print $1}' | grep -v "^#" | grep -v "^$" | grep -v "base")

echo "🧾 Base環境をバックアップ中..."
conda env export -n base > "$BACKUP_DIR/base.yaml"

# 各環境を順番にバックアップ
for env in $ENVS; do
    echo "🧾 環境 '${env}' をバックアップ中..."
    conda env export -n "$env" > "$BACKUP_DIR/${env}.yaml"
done

echo ""
echo "✅ すべての環境をバックアップしました！"
echo "📂 保存先: $BACKUP_DIR/"
ls -1 "$BACKUP_DIR"
