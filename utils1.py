
def channel_image(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
    ]

    # Create an output numpy ndarray to store the image
    if output_pixel_vals:
        out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                dtype='uint8')
    else:
        out_array = numpy.zeros((out_shape[0], out_shape[1], 4),
                                dtype=X.dtype)

    #colors default to 0, alpha defaults to 1 (opaque)
    if output_pixel_vals:
        channel_defaults = [0, 0, 0, 255]
    else:
        channel_defaults = [0., 0., 0., 1.]

    for i in xrange(4):
        if X[i] is None:
            # if channel is None, fill it with zeros of the correct
            # dtype
            dt = out_array.dtype
            if output_pixel_vals:
                dt = 'uint8'
            #out_array[:, :, i] = numpy.zeros(
            #    out_shape,
            #    dtype=dt
            #) + channel_defaults[i]
            out_array[:, :, i] = 255*numpy.ones(
                out_shape,
                dtype=dt
            ) + channel_defaults[i]
        else:
            # use a recurrent call to compute the channel and store it
            # in the output
            out_array[:, :, i] = tile_raster_images(
                X[i], img_shape, tile_shape, tile_spacing,
                scale_rows_to_unit_interval, output_pixel_vals)
    return out_array


