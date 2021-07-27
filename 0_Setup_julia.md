# julia_tutorial

- Install julia in your local environment
- Set up Jupyter lab environment for julia

## Download

https://julialang.org/downloads/

## Set PATH (for Mac)
```bash
echo "alias julia='/Applications/Julia-1.6.app/Contents/Resources/julia/bin/julia'" >> .zshrc
exec $SHELL
```

## Start julia
```bash
julia
```

## Hello world
```
print("Hello Julia!")
```

## Data handling
```
a = 1
# Data type
typeof(a)
b = "aaa"
typeof(b)
```

## Package management mode
- ] でパッケージ管理モードへ
- add package_nameでインポート

```
pkg> add Plots
```

- statusで済み	パッケージ確認
- delでパッケージモードを抜ける

## Other mode

- Juliaモード
  - Juliaのコードを実行するデフォルトのモード
- パッケージモード
  - **]** キーで起動
  - パッケージのインストールや依存関係を管理するモード
- ヘルプモード
  - **?** キーで起動
  - 関数やマクロのドキュメントを確認できます。
- シェルモード
  - **;** キーで起動
  - シェルのコマンドを実行するモード
- サーチモード
  - **Ctrl+r** で起動
  - コマンドの履歴を検索できます。

## Jupyter notebook(lab)

- anaconda でjupyter notebook or labがインストールされている前提

```
$ julia
julia> ]
# パッケージモードへ 
pkg> add IJulia
# Jupyter packageをダウンロード
julia> exit()
# julia を終了
> jupyter notebook
# jupyter notebookを起動するとjuliaのカーネルが選択可
```



