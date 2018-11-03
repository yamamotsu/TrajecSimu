# TrajecSimu

6-dof trajectory simulation for high-power rockets.  
current version: 3.0 (11/1/2018)

## 概要
Solves a 6-dof equation of motion for a trajectory of a transonic high-power rocket.  
Limited to ones without attitude/trajectory control.

Might have some problems on Windows/Linux.

## 使い方

### 必要なもの
- Python3
- Pythonライブラリ: numpy, scipy, pandas, matplotlib, numpy-quaternion(https://github.com/moble/quaternion)

### インストール

#### ZIPダウンロード
下記をクリックしてZIPダウンロード→展開
https://github.com/yamamotsu/TrajecSimu/archive/master.zip

#### gitでインストール
```sh
$ git clone https://github.com/yamamotsu/TrajecSimu
```

### サンプルコードを実行
cloneまたはダウンロードした`TrajecSimu`のフォルダをコマンドプロンプトで開き,
`python driver_sample.py`でサンプルコードを実行.

`Thrustcurve_sample`内の`sample_thrust.csv`はエンジン推力データ(スラストカーブ)のサンプル,
`Config_sample`内の`sample_config.csv`はロケットパラメータ設定ファイルのサンプルです.

## ライセンス

このソフトウェアはMITライセンスのもとで公開されています.
[MIT](https://github.com/yamamotsu/TrajecSimu/LICENSE)

## Author

[yamamotsu](https://github.com/yamamotsu)
