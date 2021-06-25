import os
import numpy as np
import h5py
import pandas as pd


def create_webpages(webpage_folder, aspis_folder, data, html_page_heading):
    """
    create webpages for given data in wgt/www folder
    :param webpage_folder:
    :param aspis_folder:
    :param data:
    :param html_page_heading:
    :return:
    """

    # create webpage
    os.makedirs(webpage_folder, exist_ok=True)
    one_row_examples = 5
    num_column = one_row_examples

    # group by level
    group_level_data = data.groupby('level')

    # create 3 pages, one for each level [1,2,3]
    for level, level_data in group_level_data:
        # reset index
        level_data = level_data.reset_index(drop=True)

        # create webpage
        with open(os.path.join(webpage_folder, f'results_page{level}.html'), 'w') as htmlfile:
            htmlfile.write("<html><head><style>table, th, td {border: 1px solid black; } </style></head>\n")
            htmlfile.write("<body>\n")

            htmlfile.write("<table style=\"width:100%\">\n")
            htmlfile.write(f"<tr><th colspan=\"{num_column}\">{html_page_heading}</th></tr>\n")
            htmlfile.write(f"<tr><th colspan=\"{num_column}\">LEVEL {level}</th></tr>\n")

            # add links to all three levels of results webpage
            #htmlfile.write("<tr>\n")
            #htmlfile.write(
            #    f"<th><a href=\"https://aspis.cmpt.sfu.ca/projects/wgt/{aspis_folder}/results_page1.html\">"
            #    f"go to Level-1</a></th>\n")
            #htmlfile.write(
            #    f"<th><a href=\"https://aspis.cmpt.sfu.ca/projects/wgt/{aspis_folder}/results_page2.html\">"
            #    f"go to Level-2</a></th>\n")
            #htmlfile.write(
            #    f"<th><a href=\"https://aspis.cmpt.sfu.ca/projects/wgt/{aspis_folder}/results_page3.html\">"
            #    f"go to Level-3</a></th>\n")
            #htmlfile.write("</tr>")

            htmlfile.write(f"<tr><th colspan=\"{num_column}\"><a href=\"#bottom\">Go to bottom</a></th></tr>")
            # group by level
            for idx, row in level_data.iterrows():
                # start a row
                if idx % one_row_examples == 0:
                    htmlfile.write("<tr>\n")
                # display images
                htmlfile.write(
                    f"<td><b style=\"font-family:courier; font-size:18px\">{row['image_idx']}</b><br>\n"
                    f"<img src=\"{row['image_name']}\" loading=\"lazy\" "
                    f"alt=\"{row['image_name']}\"></td>\n")
                # end row
                if idx % one_row_examples == (one_row_examples - 1):
                    htmlfile.write("</tr>\n")

            htmlfile.write(f"<tr><th colspan=\"{num_column}\"> <a href=\"#top\">Go to top</a></th></tr>\n")
            htmlfile.write("</table>\n")
            htmlfile.write("<h2 id=\"bottom\"></h2>\n")
            htmlfile.write("</body></html>")
        print(f"HTML file saved:{aspis_folder}/results_page{level}.html")


def create_all_results_html_page(opt_parser, htmldata, category):
    """
    create webpages for all results
    :return:
    """

    # foldername = os.path.basename(opt_parser.webpage_folder)
    foldername = opt_parser.webpage_folder.split("www")[-1][1:]
    print("foldername: ", foldername)
    webpage_folder = os.path.join(opt_parser.webpage_folder, category)
    os.makedirs(webpage_folder, exist_ok=True)
    aspis_folder = os.path.join(foldername, category)

    create_webpages(webpage_folder=webpage_folder,
                    aspis_folder=aspis_folder, data=htmldata,
                    html_page_heading=opt_parser.html_page_heading)


def get_partnet_results_data(category, rendering_folder):

    # create pandas dataframe with info required for HTML page
    html_dict = dict()
    html_dict['image_name'] = []
    html_dict['image_idx'] = []
    html_dict['level'] = []
    for level in [3]:
        # list directory
        folder = f"{rendering_folder}/{category}/Level_{level}"
        image_folder = f"../../www/{folder}"
        image_names = os.listdir(image_folder)
        # consider only 100 images
        image_count = 100
        image_names = image_names[:image_count]
        # list names
        html_dict['image_name'].extend([f"../../../{folder}/{name}" for name in image_names])
        html_dict['image_idx'].extend([i for i in range(len(image_names))])
        # save level so that we can group dataframe by this column
        html_dict['level'].extend([level for _ in range(len(image_names))])

    pd.set_option('display.max_columns', None)
    # convert dict to pandas dataframe
    htmldata = pd.DataFrame.from_dict(html_dict)
    print("htmldata:\n", htmldata.head())
    print("Number of objects:", len(htmldata))
    return htmldata


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--webpage_folder', type=str, help="folder in www where webpages will be created",
                        default='../../www/part_segmentation/pretrained')
    parser.add_argument('--category', type=str, default="Bed", help="Name of the partnet category")
    parser.add_argument("--html_page_heading", type=str,
                        default="[Pretrained model] part segmentation results for Bag",
                        help="sentence to be displayed on top of each HTML page")
    parser.add_argument('--rendering_folder', type=str, default='partnet_renderings/pretrained',
                        help="folder location where partnet part segmentation images are saved WRT where html pages are saved")

    # parser.add_argument('--num_pages', type=int, default=100, help="number of webpages to save")
    # parser.add_argument('--per_page_examples', type=int, default=50, help="number of examples per page")

    # parser.add_argument('--level', type=int, default=1,
    #                     help="integer indicating granularity of part segmentation. 1=Coarser, 3=Finer")


    opt_parser = parser.parse_args()
    # get htmldata for partnet dataset
    htmldata = get_partnet_results_data(category=opt_parser.category, rendering_folder=opt_parser.rendering_folder)

    # create results webpage
    create_all_results_html_page(opt_parser=opt_parser, htmldata=htmldata, category=opt_parser.category)
