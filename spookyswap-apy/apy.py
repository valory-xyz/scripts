from argparse import ArgumentParser
from calendar import timegm
from datetime import datetime, timedelta

import requests as requests

spooky_subgraph_url = 'https://api.thegraph.com/subgraphs/name/eerieeight/spookyswap'


def pair_query(token0: str, token1: str) -> str:
    """Create a query to get the pair's data using its tokens' ids."""
    today = datetime.now()
    today_morning = today.strftime('%Y%m%d') + '000000'
    today_morning = datetime.strptime(today_morning, '%Y%m%d%H%M%S')
    yesterday_morning = today_morning - timedelta(days=1)
    today_morning_unix = str(timegm((today_morning.utctimetuple())))
    yesterday_morning_unix = str(timegm((yesterday_morning.utctimetuple())))

    query = '''
        {
        pairDayDatas(first: 2, where: {
            token0: "''' + token0 + '''",
            token1: "''' + token1 + '''",
            date_in: [''' + today_morning_unix + ''', ''' + yesterday_morning_unix + ''']
        })
        {
            token0{symbol}
            token1{symbol}
            reserveUSD
            totalSupply
            dailyTxns
            dailyVolumeUSD
        }
    }
    '''

    return query


def create_parser() -> ArgumentParser:
    """Creates the script's argument parser."""
    parser = ArgumentParser(description='A short demo that calculates APY for a single pair.')
    parser.add_argument('token0', type=str, help="The pool pair's token0 id.")
    parser.add_argument('token1', type=str, help="The pool pair's token1 id.")

    return parser


def run_demo() -> None:
    """Runs the APY calculation demo."""
    args = create_parser().parse_args()

    r = requests.post(spooky_subgraph_url, json={'query': pair_query(args.token0, args.token1)})

    if r.status_code == 200:
        pair_result_today = r.json()['data']['pairDayDatas'][0]
        pair_result_yesterday = r.json()['data']['pairDayDatas'][1]

        token0 = pair_result_today['token0']['symbol']
        token1 = pair_result_today['token1']['symbol']

        volume_usd_today = float(pair_result_today['dailyVolumeUSD'])
        volume_usd_yesterday = float(pair_result_yesterday['dailyVolumeUSD'])
        current_change = float(volume_usd_today) - float(volume_usd_yesterday)

        reserve_usd = float(pair_result_today['reserveUSD'])

        apy = (current_change * .002 * 365 * 100) / reserve_usd
        print(f'APY for {token0}-{token1} is {apy}')
    else:
        print(f'Something went wrong while trying to communicate with the subgraph (Error: {r.status_code})!\n{r.text}')


if __name__ == '__main__':
    run_demo()
