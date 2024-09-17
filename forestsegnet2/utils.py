import sys


def multithreaded_execution(function, parameters, threads=20):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import inspect
    import tqdm

    # Get parameter list
    fun_params = inspect.signature(function).parameters.keys()
    fun_progressbar_param = True if "progressbar" in fun_params else False

    n_items = len(parameters)
    assert n_items > 0, "Empty list of parameters passed."
    print("\n", "Processing {:,d} images".format(n_items))

    with tqdm.tqdm(
        total=n_items, bar_format="{l_bar}{bar:75}{r_bar}{bar:-50b}", file=sys.stdout
    ) as pbar:
        if fun_progressbar_param:
            _ = [p.update({"progressbar": pbar}) for p in parameters]

        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = [executor.submit(function, **param) for param in parameters]
            results = []

            try:  # catch exceptions
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)

            except Exception as e:
                print(f'Exception "{e}" raised while processing files.')
                raise e

        return results
