"""Default graphical collector; legacy key-only collector is optional."""
import bootstrap
import argparse
import sys
from lg.labels import LABELS


def main():
    if '--keyboard' in sys.argv:
        sys.argv.remove('--keyboard')
        from collect_digits_keyboard import main as keyboard_main
        return keyboard_main()
    parser=argparse.ArgumentParser(description='YTO 手势按钮采集工具')
    parser.add_argument('--operator',default='p01')
    parser.add_argument('--session',default=None)
    parser.add_argument('--camera',default='1')
    parser.add_argument('--label',default='1',choices=LABELS)
    parser.add_argument('--hand',choices=['left','right'])
    parser.add_argument('--side',choices=['left','right','both'],default='both')
    parser.add_argument('--demo',action='store_true',help='合成输入；数据单独保存，不写正式数据集')
    parser.add_argument('--smoke',type=float,default=0)
    args=parser.parse_args()
    from lg.collector import CollectorApp
    CollectorApp(args,bootstrap.ROOT).run()


if __name__=='__main__': main()
