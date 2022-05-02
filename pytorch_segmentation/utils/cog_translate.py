import rasterio
from rasterio.enums import Resampling as ResamplingEnums
from rasterio.vrt import WarpedVRT
from rasterio.rio.overview import get_maximum_overview_level
from rasterio.shutil import copy


def cog_translate(dataset,out_file,out_meta,resampling="nearest"):
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True,GDAL_TIFF_OVR_BLOCKSIZE=128):
        out_meta = out_meta.copy()
        tilesize = min(int(out_meta["blockxsize"]), int(out_meta["blockysize"]))

        overview_level = get_maximum_overview_level(
                            dataset.width, dataset.height, minsize=tilesize
                        )

        overviews = [2**j for j in range(1, overview_level + 1)]
        dataset.build_overviews(overviews, ResamplingEnums[resampling])

        tags = dataset.tags()
        tags.update(
            dict(
                OVR_RESAMPLING_ALG=ResamplingEnums[
                    resampling
                ].name.upper()
            )
        )

        dataset.update_tags(**tags)
        out_meta["driver"] = "COG"
        out_meta["overview_resampling"] = resampling
        out_meta["warp_resampling"] = resampling
        out_meta["blocksize"] = tilesize
        out_meta.pop("blockxsize", None)
        out_meta.pop("blockysize", None)
        out_meta.pop("tiled", None)
        copy(dataset, out_file, **out_meta)


# def cog_translate(dataset,out_file,out_meta,resampling="nearest"):
#     out_meta = out_meta.copy()
#     tilesize = min(int(out_meta["blockxsize"]), int(out_meta["blockysize"]))
#     vrt_params = {
#         "add_alpha": False,
#         "dtype": dataset.meta["dtype"],
#         "width": dataset.width,
#         "height": dataset.height,
#         "resampling": ResamplingEnums[resampling],
#         "src_nodata": dataset.meta["nodata"],
#         "nodata": dataset.meta["nodata"]
#     }

#     with WarpedVRT(dataset, **vrt_params) as vrt_dst:
#         # meta = vrt_dst.meta
#         # meta.update({"compress":out_meta["compress"],"count":1})

#         with rasterio.MemoryFile() as tmpfile:
#             with tmpfile.open(**out_meta) as tmp_dst:
#                 # print(out_meta)
#                 # print(tmp_dst.meta)
#                 # print(vrt_dst.meta)
#                 for _, win in tmp_dst.block_windows(1):
#                     arr = vrt_dst.read(window=win,indexes=1)
#                     tmp_dst.write(arr, window=win,indexes=1)

#                 overview_level = get_maximum_overview_level(
#                         vrt_dst.width, vrt_dst.height, minsize=tilesize
#                     )

#                 overviews = [2**j for j in range(1, overview_level + 1)]
#                 tmp_dst.build_overviews(overviews, ResamplingEnums[resampling])

#                 tags = dataset.tags()
#                 tags.update(
#                     dict(
#                         OVR_RESAMPLING_ALG=ResamplingEnums[
#                             resampling
#                         ].name.upper()
#                     )
#                 )

#                 tmp_dst.update_tags(**tags)
#                 out_meta["driver"] = "COG"
#                 out_meta["overview_resampling"] = resampling
#                 out_meta["warp_resampling"] = resampling
#                 out_meta["blocksize"] = tilesize
#                 out_meta.pop("blockxsize", None)
#                 out_meta.pop("blockysize", None)
#                 out_meta.pop("tiled", None)
#                 copy(tmp_dst, out_file, **out_meta)
        

