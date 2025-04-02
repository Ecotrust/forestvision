"""OSU GNN Forest Attributes dataset module.

This module provides the GNNForestAttr class for accessing Oregon State University's
Gradient Nearest Neighbor (GNN) forest attributes data from 2017, which includes
various forest structure and composition metrics.
"""

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy
from rasterio.crs import CRS
from torchgeo.datasets import RasterDataset
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class GNNForestAttr(RasterDataset):
    """OSU GNN Forest Attributes 2017.

    This dataset provides access to Oregon State University's Gradient Nearest Neighbor
    (GNN) forest attributes data from 2017, including forest structure, composition,
    and biomass metrics.

    Attributes:
        _res (int): Internal resolution storage (30 meters).
        is_image (bool): Flag indicating this dataset contains mask data, not image data.
        filename_glob (str): Pattern for matching dataset files.
        filename_regex (str): Regular expression pattern for parsing filenames.
        separate_files (bool): Flag indicating bands are stored in separate files.
        nodata (int): NoData value for the dataset.
        all_bands (List[str]): List of all available dataset bands.
        remap_dict (Dict[int, int]): Dictionary for remapping forest type codes.

    Bands (stored as separate rasters):
        - fortypba: forest type
        - cancov: canopy cover from 0 to 10,000
        - stndhgt: height of dominant and co-dominant trees in cm
        - mndbhba: basal-area-weighted average dbh of live trees, in mm
        - ba_ge_3: basal area of live trees >2.5cm dbh, m2/ha
        - tph_ge_3: live trees per hectare >2.5cm dbh
        - bph_ge_3_crm: biomass of live trees >2.5cm dbh, kg/ha
        - cancov_layers: number of canopy cover layers

    Attribution:
        Landscape Ecology Modeling, Mapping, and Analysis (LEMMA) Team. 2020. Gradient Nearest
        Neighbor (GNN) raster dataset (version 2020.01). Modeled forest vegetation data using
        direct gradient analysis and nearest neighbor imputation.

    Reference:
        Retrieved from: https://lemmadownload.forestry.oregonstate.edu/

    Source CRS: EPSG:5070
    Spatial Resolution: 30 meters
    """

    _res = 30
    is_image = False
    filename_glob = "*.tif"
    filename_regex = r"^(?P<band>\w+)_2017.tif$"
    separate_files = True
    nodata = -2147483648
    all_bands = [
        "fortypba",
        "cancov",
        "stndhgt",
        "mndbhba",
        "ba_ge_3",
        "tph_ge_3",
        "bph_ge_3_crm",
        "cancov_layers",
    ]

    # fmt: off
    # GNN --> ODF code mapping
    remap_dict = {
         -1: -1,   1:  0,   2:  6,   3:  7,   4:  7,   5:  7,   6:  9,   7:  2,   8:  2,   9: 12,  10: 12,  11: 12,  12: 13,  13: 12,  14: 12,
         15: 12,  16: 12,  17: 12,  18: 12,  19: 12,  20: 12,  21: 12,  22: 12,  23: 12,  24: 12,  25: 12,  26: 12,  27:  5,  28:  5,  29: 12,
         30: 13,  31: 12,  32:  5,  33:  9,  34:  9,  35: 11,  36:  9,  37:  9,  38:  9,  39:  5,  40:  5,  41:  5,  42: 10,  43: 11,  44:  5,
         45:  5,  46:  5,  47: 13,  48:  5,  49:  5,  50:  5,  51:  5,  52: 13,  53: 13,  54:  5,  55:  5,  56:  5,  57:  5,  58:  9,  59:  5,
         60:  9,  61:  5,  62:  9,  63:  7,  64:  7,  65:  2,  66: 12,  67: 11,  68: 11,  69: 11,  70: 12,  71:  5,  72: 13,  73: 13,  74: 13,
         75: 13,  76: 13,  77: 13,  78: 13,  79: 13,  80: 13,  81: 13,  82: 13,  83: 13,  84: 13,  85: 13,  86: 13,  87: 13,  88:  5,  89: 12,
         90:  5,  91: 13,  92:  5,  93: 12,  94:  5,  95:  5,  96:  5,  97: 12,  98: 13,  99:  5, 100:  5, 101:  5, 102:  5, 103: 12, 104: 12,
        105: 12, 106: 12, 107: 12, 108:  9, 109:  5, 110:  4, 111:  9, 112:  9, 113:  9, 114:  9, 115:  9, 116:  9, 117:  9, 118:  9, 119:  9,
        120:  9, 121:  5, 122:  9, 123: 10, 124:  9, 125:  9, 126:  9, 127:  9, 128:  9, 129:  9, 130:  9, 131:  9, 132:  9, 133:  9, 134:  9,
        135:  9, 136:  9, 137:  5, 138:  2, 139:  5, 140:  7, 141:  7, 142:  7, 143:  7, 144:  7, 145:  7, 146:  5, 147:  2, 148:  9, 149:  5,
        150:  5, 151:  5, 152:  2, 153:  2, 154:  7, 155:  7, 156:  7, 157:  7, 158:  7, 159:  2, 160:  2, 161: 12, 162:  2, 163:  2, 164:  2,
        165:  9, 166:  2, 167:  2, 168:  2, 169:  2, 170: 10, 171:  2, 172:  2, 173:  5, 174:  2, 175:  2, 176:  2, 177: 10, 178:  2, 179:  2,
        180:  2, 181:  2, 182: 11, 183:  2, 184: 10, 185:  2, 186: 11, 187:  2, 188:  9, 189:  9, 190:  9, 191:  9, 192:  9, 193:  9, 194:  5,
        195:  5, 196:  9, 197:  9, 198:  9, 199:  9, 200:  9, 201:  9, 202:  9, 203:  9, 204:  9, 205:  9, 206:  9, 207:  9, 208:  2, 209:  2,
        210:  9, 211:  9, 212:  5, 213:  2, 214:  2, 215:  9, 216:  5, 217:  5, 218:  9, 219:  9, 220:  9, 221:  9, 222:  5, 223:  5, 224:  5,
        225:  5, 226:  5, 227:  5, 228:  5, 229:  5, 230:  5, 231:  9, 232:  5, 233:  5, 234:  9, 235:  9, 236:  7, 237:  7, 238:  9, 239:  5,
        240:  5, 241:  1, 242:  5, 243:  1, 244:  6, 245: 13, 246:  5, 247:  5, 248:  4, 249:  2, 250:  5, 251:  7, 252:  1, 253:  5, 254:  9,
        255:  5, 256:  9, 257:  5, 258:  5, 259:  9, 260:  9, 261: 10, 262: 10, 263: 10, 264:  5, 265: 10, 266: 10, 267:  5, 268:  5, 269: 10,
        270: 10, 271: 10, 272: 10, 273: 13, 274: 12, 275:  5, 276: 13, 277: 12, 278: 13, 279: 13, 280: 12, 281:  2, 282:  9, 283:  9, 284:  9,
        285: 10, 286: 10, 287:  5, 288:  5, 289:  5, 290:  2, 291:  9, 292:  2, 293:  9, 294:  2, 295:  2, 296:  2, 297:  7, 298:  2, 299:  6,
        300:  6, 301:  5, 302:  5, 303:  5, 304:  7, 305:  6, 306:  9, 307:  7, 308:  6, 309:  5, 310:  5, 311:  6, 312:  5, 313:  6, 314:  3,
        315:  5, 316:  5, 317:  5, 318:  4, 319:  9, 320:  2, 321:  5, 322:  9, 323:  7, 324:  7, 325:  7, 326:  6, 327:  1, 328:  5, 329:  5,
        330: 12, 331: 13, 332: 13, 333: 13, 334: 12, 335:  5, 336:  5, 337:  5, 338: 13, 339:  5, 340:  5, 341:  5, 342:  5, 343: 13, 344:  5,
        345:  5, 346:  9, 347:  5, 348:  5, 349:  5, 350:  5, 351:  5, 352:  5, 353:  5, 354:  5, 355:  5, 356:  5, 357:  5, 358:  5, 359:  5,
        360:  5, 361:  5, 362:  5, 363:  5, 364:  5, 365:  5, 366:  5, 367:  5, 368: 10, 369:  5, 370:  5, 371:  7, 372:  0, 373:  1, 374: 13,
        375: 13, 376: 13, 377: 13, 378: 13, 379: 13, 380: 13, 381: 13, 382: 13, 383:  5, 384:  5, 385:  5, 386:  5, 387:  5, 388:  5, 389:  5,
        390:  5, 391:  5, 392:  5, 393:  5, 394:  5, 395:  5, 396:  5, 397:  5, 398:  5, 399:  5, 400:  5, 401:  5, 402:  5, 403:  3, 404: 12,
        405:  5, 406: 13, 407: 12, 408:  3, 409:  3, 410:  3, 411:  5, 412:  3, 413:  5, 414:  5, 415: 13, 416:  5, 417:  5, 418: 13, 419: 13,
        420:  5, 421:  5, 422:  5, 423:  5, 424:  5, 425: 10, 426:  9, 427:  9, 428:  5, 429:  3, 430:  5, 431: 12, 432:  5, 433:  5, 434:  5,
        435:  5, 436:  5, 437: 13, 438: 13, 439: 13, 440: 13, 441: 13, 442: 13, 443: 13, 444: 13, 445: 13, 446: 13, 447: 13, 448: 13, 449: 13,
        450: 13, 451: 13, 452: 13, 453: 13, 454: 13, 455: 13, 456: 13, 457: 13, 458: 13, 459: 13, 460:  5, 461:  5, 462:  5, 463:  5, 464:  5,
        465:  5, 466:  5, 467:  5, 468:  5, 469:  5, 470:  5, 471:  5, 472:  5, 473:  5, 474:  5, 475:  5, 476:  5, 477:  5, 478:  5, 479:  5,
        480:  5, 481:  5, 482:  5, 483:  5, 484:  5, 485:  5, 486:  5, 487:  5, 488:  9, 489:  5, 490:  5, 491:  5, 492:  5, 493:  5, 494:  5,
        495:  5, 496:  5, 497:  5, 498:  9, 499:  7, 500: 13, 501:  5, 502:  5, 503:  5, 504:  5, 505:  5, 506:  5, 507:  5, 508:  5, 509:  5,
        510:  5, 511:  5, 512:  5, 513:  5, 514: 12, 515:  5, 516: 12, 517: 12, 518: 10, 519: 13, 520:  5, 521:  5, 522:  5, 523: 13, 524:  5,
        525:  5, 526:  5, 527:  5, 528:  5, 529:  5, 530: 12, 531:  5, 532:  5, 533:  5, 534:  5, 535: 10, 536:  4, 537:  4, 538:  4, 539:  5,
        540:  5, 541:  5, 542:  4, 543:  9, 544:  4, 545:  9, 546:  9, 547:  4, 548:  5, 549:  4, 550:  4, 551:  4, 552:  2, 553:  4, 554:  4,
        555:  5, 556:  5, 557:  5, 558:  5, 559:  5, 560: 13, 561:  5, 562:  5, 563:  5, 564:  5, 565:  9, 566:  9, 567:  4, 568:  9, 569:  9,
        570:  5, 571:  9, 572:  5, 573:  7, 574:  7, 575:  7, 576:  7, 577:  4, 578:  5, 579:  5, 580: 10, 581: 10, 582:  5, 583:  5, 584:  5,
        585:  5, 586:  5, 587:  5, 588:  5, 589:  5, 590:  5, 591:  5, 592:  5, 593:  5, 594:  5, 595:  5, 596:  5, 597: 10, 598: 10, 599: 10,
        600: 10, 601: 10, 602: 10, 603: 10, 604: 10, 605: 10, 606: 10, 607: 10, 608: 10, 609:  5, 610:  9, 611:  5, 612:  5, 613:  2, 614:  9,
        615:  2, 616:  2, 617:  2, 618:  2, 619:  9, 620:  2, 621:  9, 622:  9, 623: 13, 624:  9, 625:  9, 626:  7, 627:  7, 628:  2, 629:  2,
        630:  2, 631:  2, 632:  8, 633:  8, 634:  9, 635:  2, 636:  2, 637:  8, 638:  8, 639:  8, 640:  8, 641:  8, 642:  2, 643:  8, 644:  1,
        645:  9, 646: 12, 647:  9, 648:  1, 649:  1, 650:  1, 651:  2, 652:  1, 653:  9, 654: 11, 655:  2, 656:  5, 657:  5, 658:  5, 659:  7,
        660:  7, 661: 11, 662: 11, 663: 12, 664:  5, 665: 13, 666: 12, 667: 11, 668:  9, 669:  9, 670: 11, 671:  5, 672: 11, 673: 11, 674:  9,
        675:  5, 676: 11, 677: 11, 678:  5, 679: 11, 680:  5, 681: 10, 682: 13, 683: 11, 684: 11, 685: 11, 686:  5, 687:  5, 688:  5, 689: 11,
        690: 11, 691: 13, 692:  5, 693:  5, 694:  5, 695: 13, 696:  5, 697:  5, 698: 11, 699:  5, 700:  5, 701: 10, 702:  9, 703: 11, 704: 11,
        705: 11, 706:  9, 707:  7, 708:  9, 709:  5, 710:  7, 711:  7, 712: 11, 713:  5, 714: 11, 715:  2, 716: 11, 717: 10, 718: 11, 719: 11,
        720: 11, 721: 11, 722: 12, 723: 11, 724:  7, 725:  7, 726:  9, 727:  7, 728:  9, 729:  7, 730:  7, 731:  5, 732:  7, 733:  7, 734:  7,
        735:  7, 736:  7, 737:  7, 738:  7, 739:  7, 740:  7, 741:  7, 742:  7, 743: 10, 744:  7, 745:  7, 746:  7, 747:  7, 748:  9, 749:  7,
        750:  7, 751:  7, 752:  9, 753:  7, 754:  7, 755:  7, 756:  7, 757:  5, 758:  5, 759:  7, 760:  5, 761:  7, 762:  7, 763:  7, 764:  5,
        765:  7, 766:  7, 767:  9, 768:  7, 769:  7, 770:  7, 771:  7, 772:  7, 773:  7, 774:  7, 775:  7, 776:  9, 777:  7, 778:  7, 779:  5,
        780:  7, 781:  7, 782:  7, 783:  7, 784:  7, 785:  7, 786:  7, 787:  7, 788:  7, 789:  7, 790:  7, 791:  9, 792:  7, 793:  7, 794:  9,
        795:  7, 796:  7, 797:  7, 798:  7, 799:  7, 800:  7, 801:  5, 802:  7, 803:  7, 804:  7, 805:  7, 806:  7, 807:  7, 808:  7, 809:  7,
        810:  7, 811:  7, 812:  7, 813:  7, 814:  7, 815:  9, 816:  7, 817:  7, 818:  9, 819:  7, 820:  5, 821:  5, 822:  5, 823:  7, 824:  7,
        825:  5, 826:  7, 827:  7, 828:  7, 829:  7, 830:  7, 831:  7, 832:  7, 833:  7, 834:  7, 835:  7, 836:  9, 837:  7, 838:  9, 839:  9,
        840:  9, 841:  9, 842:  5, 843:  7, 844:  9, 845:  7, 846:  7, 847:  7, 848:  7, 849:  7, 850:  7, 851:  7, 852:  9, 853:  7, 854:  7,
        855:  9, 856:  7, 857:  5, 858:  5, 859:  7, 860:  7, 861:  5, 862:  7, 863:  7, 864:  7, 865:  7, 866:  7, 867:  7, 868:  7, 869:  7,
        870:  7, 871:  7, 872:  7, 873:  7, 874:  1, 875:  2, 876:  2, 877:  2, 878:  2, 879:  2, 880:  7, 881:  5, 882:  5, 883:  5, 884:  5,
        885:  5, 886: 10, 887: 10, 888: 10, 889: 10, 890: 10, 891: 10, 892: 10, 893: 10, 894: 10, 895: 10, 896: 10, 897: 10, 898: 10, 899: 10,
        900: 10, 901: 11, 902: 11, 903: 12, 904: 11, 905: 13, 906: 11, 907:  9, 908: 11, 909: 11, 910: 11, 911: 11, 912:  5, 913:  5, 914: 13,
        915: 11, 916: 10, 917:  9, 918: 11, 919: 11, 920:  7, 921: 11, 922: 13, 923:  5, 924: 11, 925: 12, 926: 11, 927: 13, 928: 12, 929:  9,
        930: 11, 931: 11, 932: 11, 933:  5, 934:  5, 935: 10, 936: 13, 937:  5, 938:  5, 939: 13, 940:  5, 941:  5, 942: 10, 943: 11, 944: 11,
        945: 11, 946: 10, 947: 11, 948: 11, 949: 12, 950: 12, 951: 12, 952: 12, 953: 13, 954: 12, 955: 12, 956: 13, 957:  5, 958: 12, 959: 13, 
        960: 12, 961: 12, 962: 12, 963: 12, 964: 12, 965:  5, 966:  9, 967:  2, 968:  9, 969: 10, 970:  2, 971:  2, 972:  5, 973:  5, 974:  5, 
        975:  9, 976:  7, 977:  7, 978:  7, 979:  7, 980:  7, 981:  7, 982: 10, 983:  9
    }
    # fmt: on

    def __init__(
        self,
        paths: Path | Iterable[Path] = "data/datasets/osugnn",
        bands: Sequence[str] = ["fortypba"],
        remap: bool = True,
        crs: CRS | None = None,
        res: float | None = 30,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = False,
    ) -> None:
        """Initialize OSU GNN Forest Structure and Composition dataset.

        Args:
            paths (Path | Iterable[Path]): A directory containing the dataset files
                or a list of paths. Defaults to "data/datasets/osugnn".
            bands (Sequence[str], optional): List of bands to load from the dataset.
                Defaults to ["fortypba"].
            remap (bool, optional): Whether to remap forest type codes using the
                built-in remapping dictionary. Defaults to True.
            crs (CRS, optional): Optional coordinate reference system to reproject
                the dataset to.
            res (float, optional): Optional resolution to resample the dataset to.
                Defaults to 30 meters.
            transforms (Callable, optional): Optional function/transform to apply
                to each sample.
            cache (bool, optional): Flag indicating whether to cache the dataset
                in memory. Defaults to False.
        """
        self.paths = paths
        if res:
            self._res = res
        self.remap = remap

        super().__init__(
            paths, crs, res, bands=bands, transforms=transforms, cache=cache
        )

    def remap_fortypba(self, a) -> Any:
        """Remap forest type codes using the built-in remapping dictionary.

        Args:
            a (Any): Array containing forest type codes to be remapped.

        Returns:
            Any: Array with remapped forest type codes.

        Note:
            Uses the remap_dict attribute to convert GNN forest type codes to
            ODF (Oregon Department of Forestry) standard codes.
        """
        for k in a.unique():
            k = int(k)
            a[a == k] = self.remap_dict.get(k, -1)
        return a

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a sample from the dataset.

        Args:
            idx (int): The index of the sample to return.

        Returns:
            dict[str, Any]: A dictionary containing the sample data with keys
                'mask' (and optionally 'prediction' if available).

        Note:
            If remap is enabled and 'fortypba' band is selected, forest type codes
            will be remapped using the built-in remapping dictionary.
        """
        sample = super().__getitem__(idx)

        if self.remap and "fortypba" in self.bands:
            fortypba_idx = self.bands.index("fortypba")
            fortypba = sample["mask"][fortypba_idx]
            sample["mask"][fortypba_idx] = self.remap_fortypba(fortypba)
        return sample

    def plot(
        self,
        sample: dict[str, Any],
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample (dict[str, Any]): A sample returned by :meth:`RasterDataset.__getitem__`
                containing at least a 'mask' key, and optionally a 'prediction' key.
            show_titles (bool, optional): Flag indicating whether to show titles
                above each panel. Defaults to True.
            suptitle (str, optional): Optional string to use as a suptitle for
                the entire figure.

        Returns:
            Figure: A matplotlib Figure with the rendered sample.

        Note:
            If the sample contains a 'prediction' key, both mask and prediction
            will be plotted side by side. Otherwise, only the mask is displayed.
        """
        mask = sample["mask"].squeeze()
        ncols = 1

        showing_predictions = "prediction" in sample
        if showing_predictions:
            pred = sample["prediction"].squeeze()
            ncols = 2

        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 4, 4))

        if showing_predictions:
            axs[0].imshow(mask)
            axs[0].axis("off")
            axs[1].imshow(pred)
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Mask")
                axs[1].set_title("Prediction")
        else:
            axs.imshow(mask)
            axs.axis("off")
            if show_titles:
                axs.set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
