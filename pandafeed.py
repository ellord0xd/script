class PandasData(bt.feeds.PandasData):
    params = (
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('datetime', 'Datetime'),
        ('dtformat', '%Y-%m-%d %H:%M:%S'),
        ('timeframe', bt.TimeFrame.Minutes),
        ('compression', 1),
        ('fromdate', None),
        ('todate', None),
        ('reverse', False),
    )

def __init__(self, dataname, **kwargs):
    self._data = pd.DataFrame(dataname)
    self._data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    self._data['datetime'] = pd.to_datetime(self._data['datetime'], unit='ms')
    self._data.set_index('datetime', inplace=True)
    super().__init__(**kwargs)
``This modified `PandasData` class checks if the `dataname` parameter is a list, and converts it to a pandas dataframe with the expected column names if it is. You can try replacing the `PandasData` class in your `pandafeed.py` file with this modified version and see if it resolves the issue.
