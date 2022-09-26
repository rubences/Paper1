import inspect
import re
import unittest
import math
import pandas as pd
import código as co
from datetime import datetime, timedelta


class TestCornYear(unittest.TestCase):

  inicio = input('Teclear fecha inicial deseada (dd/mm/yyyy): ')
  final = input('Teclear fecha final deseada (dd/mm/yyyy): ')

  F_inicio = datetime.strptime(inicio, "%d/%m/%Y")
  F_final = datetime.strptime(final, "%d/%m/%Y")

  def __init__(self, year):
        # Define initialization method:
      self.year = year
      if not isinstance(self.year, (int, float)):
          raise TypeError("year must be a number")
      elif not 2017 >= self.year >= 2013:
          raise ValueError("year must be between 2013 and 2017 inclusive")

  def year_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month
  print(year_month(F_final,F_inicio))

if __name__ == '__main__':
    fptr = open('output.txt', 'w')

    runner = unittest.TextCornYear(fptr)

    unittest.main(testRunner=runner, exit=False)

    fptr.close()

    with open('output.txt') as fp:
        output_lines = fp.readlines()

    pass_count = [len(re.findall(r'\.', line)) for line in output_lines if line.startswith('.')
                  and line.endswith('.\n')]

    pass_count = pass_count[0]

    print(str(pass_count))

    doc1 = inspect.getsource(TestCornYear.corn_OHLC2013-2017.txt)
    doc2 = inspect.getsource(TestCornYear.corn2013-2017.txt)
    doc3 = inspect.getsource(TestCornYear.corn2015-2017.txt)

    assert1_count = len(re.findall(r'assertEqual', doc1))

    print(str(assert1_count))

    assert1_count = len(re.findall(r'assertEqual', doc2))

    print(str(assert1_count))

    assert1_count = len(re.findall(r'assertEqual', doc3))

    print(str(assert1_count))