import argparse, os.path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='plot the file')
parser.add_argument('datfile', metavar='datfile', type=str, nargs='+', help='input file')
parser.add_argument('--xlog', dest='xlog', action='store_true', help='log scale on X')
parser.add_argument('--ylog', dest='ylog', action='store_true', help='log scale on Y')
parser.add_argument('--xlabel', dest='xlabel', type=str, default='', help='label on X')
parser.add_argument('--ylabel', dest='ylabel', type=str, default='', help='label on Y')
parser.add_argument('--delm', dest='delimiter', type=str, default=' ', help='delimiter')
parser.add_argument('-l', '--legend', dest='legend', action='store_true', help='legend')
parser.add_argument('-m', '--marker', dest='marker', type=str, default='o', help='delimiter')
parser.add_argument('--ls', dest='linestyle', type=str, default='-', help='linestyle')
parser.add_argument('-c', '--columns', dest='columns', type=str, default='0,1', help='column index')
parser.add_argument('-f', '--format', dest='format', type=str, default='', help='output file format')
parser.add_argument('-o', dest='output', type=str, default='', help='output file name')

args = parser.parse_args()

names = tuple(args.columns.split(','))
cols = tuple([int(x) for x in names])
for infile in args.datfile:
    label = "%s%s" % (os.path.basename(infile), cols)
    plt.plotfile(infile,
            cols=cols, names=names,
            delimiter=args.delimiter, marker=args.marker, linestyle=args.linestyle,
            label=label)
if args.xlabel:
    plt.xlabel(args.xlabel)
if args.ylabel:
    plt.ylabel(args.ylabel)
if args.xlog:
    plt.xscale('log')
if args.ylog:
    plt.yscale('log')
if args.legend:
    plt.legend(loc='best')
if args.output:
    plt.savefig(args.output)
    print("file %s was written" % args.output)
else:
    if args.format:
        root, ext = os.path.splitext(args.datfile[0])
        outputfile = os.path.basename(root) + '.' + args.format
        plt.savefig( outputfile )
        print("file %s was written" % outputfile)
    else:
        plt.show()

