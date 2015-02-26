from countour_finder import ContourFinder

__author__ = 'Sam Davies'


class CountSymbols(ContourFinder):

    def __init__(self, img):
        super(CountSymbols, self).__init__(img)

        self.symbol_count = len(self.symbol_contours)
        print "------"

