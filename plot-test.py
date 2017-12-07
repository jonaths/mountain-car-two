slices = 4
for s in range(slices):


    # Plot
    bin = '{0:04b}'.format(s)
    print type(bin)
    print bin, bin[-2], bin[-1]
    # print bin(s)[1], bin(s)[0]
    # print 1 if bin(s)[1] else 0, 1 if bin(s)[0] else 0
