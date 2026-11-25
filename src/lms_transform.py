"""Container for one LMS slice distortion transform and its FITS serialisation.

Purpose
-------
A :class:`Transform` holds the four ``svd_order x svd_order`` (4 x 4) matrices
``a, b, ai, bi`` that map, for one slice and echelle order, between the LMS
entrance focal plane (EFP) and the detector mosaic focal plane (MFP), together
with the instrument configuration they belong to.  The module also owns the
per-detector affine matrices (``Transform.affines``) that map MFP millimetres
to detector pixel coordinates (DFP), and the code that writes/reads a slice
transform as a FITS binary-table extension.  The polynomial form is that of
E-TNT-ATC-MET-1058 (see ``lmsdist_util.Util.apply_svd_distortion``).

Program
-------
``lmsdist`` (transforms are created in ``lmsdist_trace.RayTrace.create_svd_transforms``
and written by ``lms_filer.Filer.write_svd_transforms``); also read by
``lmssim`` and ``lmsaiv`` (LMS-OPT-02).

Coordinate systems
------------------
- EFP: ``efp_x`` mm along slice (alpha), ``efp_y`` mm across slice (beta),
  ``efp_w`` wavelength in microns.  Wavelength enters the polynomials as the
  phase ``phi = n * lambda / (2 d)`` (``lmsdist_util.Util.waves_to_phases``).
- MFP: ``mfp_x``, ``mfp_y`` mm in the 2 x 2 detector mosaic plane, origin at the
  mosaic centre.  ``mfp_x`` increases towards shorter wavelength.
- DFP: ``det_no`` (1..4), ``dfp_x`` column, ``dfp_y`` row in pixels.

Configuration dictionaries
--------------------------
``lms_configuration`` follows ``Globals.lms_config_template``
(``opticon, pri_ang, ech_ang``) and ``slice_configuration`` follows
``Globals.slice_config_template`` (``slice_no, ech_ord, w_min, w_max``).
``slice_no`` is the encoded ``100 * pslice_no + fslice_no`` universal slice
number (``lmsdist_util.Util.encode_slice_no``).

FITS product (written by ``Filer.write_svd_transforms``)
--------------------------------------------------------
File name ``lms_efp_mfp{_nom|_ext}_pa#####_ea[p|n]#####_v<date>.fits`` in
``../output/distortion/<opticon>/<date>/fits/`` relative to ``src/``, where the
tag is ``Globals.make_lms_config_id`` (``int(10000 * angle)``, zero padded).
The original docstring of the writer says "This is the data product we provide
to ScopeSim and the pipeline".

- Primary HDU (:meth:`Transform.make_hdu_primary`): header cards ``OPTICON``
  (str), ``PRI_ANG`` (deg), ``ECH_ANG`` (deg), ``N_MATS = 4``, ``MAT_ORD = 4``,
  followed by an empty card with keyword ``W``.
- One ``BinTableHDU`` per slice/order (:meth:`Transform.make_hdu_ext`),
  ``EXTNAME = 'SLICE_<slice_no>'``, header cards ``SLICE_NO``, ``ECH_ORD``,
  ``W_MIN``, ``W_MAX`` (microns), and four columns ``A, B, AI, BI`` each of 16
  rows holding the row-major flattened 4 x 4 matrix, TFORM ``'E'`` (float32).

Notes
-----
- Inconsistency: :meth:`Transform.create_affines` uses ``Globals.det_gap``
  (3.0 mm) for the inter-detector gap, whereas ``lms_detector.Detector.mosaic_gap``
  is 2.23 mm (and the irdb ``FPA_metis_lms_layout.dat`` implies 2.23 mm from
  ``x_cen = +-19.547 mm``).  The affines therefore place the detectors 0.77 mm
  further apart than ``Detector`` does.
- Clean-up: matrix columns are written with FITS format ``'E'`` (float32);
  ``'D'`` (float64) should be used so cubic terms keep 0.1 px precision over a
  2048 px detector.
- Clean-up: the EFP <-> MFP <-> DFP mapping functions live in
  ``lmsdist_util.Util`` (``efp_to_mfp``, ``mfp_to_dfp`` ...), not on this class,
  although they operate exclusively on ``Transform`` objects and
  ``Transform.affines``.
- Clean-up: ``Transform.affines`` is a class-level singleton created
  implicitly the first time any ``Transform`` is instantiated.
- Clean-up: :meth:`Transform.read_pickle` / :meth:`Transform.write_pickle`
  duplicate ``lms_filer.Filer.read_pickle`` / ``write_pickle`` and are unused.
- Clean-up: :meth:`Transform._get_results_path` and :meth:`Transform._make_fits_tag`
  are dead code (the former references ``self.output_folder`` and
  ``self.get_folder`` which do not exist on this class).
- Open question: in extended mode several extensions may share the same
  ``EXTNAME`` (same ``slice_no``, different ``ech_ord``) if a spectral slice
  receives more than one order; ``Filer.read_svd_transforms`` reads by index so
  this is harmless there, but name-based access would be ambiguous.
"""
import math
import pickle
from astropy.io import fits
from astropy.io.fits import Card
import numpy as np
from lms_globals import Globals


class Transform:
    """One slice/order EFP <-> MFP polynomial transform plus the shared affines.

    Attributes
    ----------
    wavelength_bounds : tuple of float
        ``(w_min, w_max)`` of the rays used in the fit (microns).  Set only by
        ``RayTrace.create_svd_transforms`` (no longer initialised in
        ``__init__``, so transforms read from FITS lack the attribute);
        distinct from the back-projected ``slice_configuration['w_min'/'w_max']``.
    matrices : dict
        Keys ``'a', 'b', 'ai', 'bi'`` -> ``(4, 4)`` ``numpy.ndarray`` (or
        ``None`` until fitted / read).  ``a, b`` give ``(phase, alpha) ->
        (mfp_x, mfp_y)``; ``ai, bi`` give ``(mfp_x, mfp_y) -> (phase, alpha)``.
    lms_configuration : dict
        ``opticon, pri_ang, ech_ang`` (``Globals.lms_config_template``).
    slice_configuration : dict
        ``slice_no, ech_ord, w_min, w_max`` (``Globals.slice_config_template``).
    kw_comments : dict (class attribute)
        FITS header comments keyed by configuration key.
    affines : dict or None (class attribute)
        ``{'m': (4, 3, 3), 'mi': (4, 3, 3)}`` MFP -> DFP and DFP -> MFP affine
        matrices, one per detector, created by :meth:`create_affines`.

    Notes
    -----
    - Clean-up: ``kw_comments`` has no entry for ``'spifu_no'`` although
      :meth:`make_hdu_primary` filters that key; the key never occurs in the
      templates.
    """

    kw_comments = {'opticon': 'LMS wavelength coverage (nominal or extended)',
                   'slice_no': 'LMS unique slice number',
                   'w_min': 'Shortest slice wavelength',
                   'w_max': 'Longest slice wavelength',
                   'pri_ang': 'Prism rotation angle / deg.',
                   'ech_ang': 'Echelle rotation angle / deg.',
                   'ech_ord': 'Echelle diffraction order',
                   'n_mats': 'No of matrices (A, B, AI, BI)',
                   'mat_ord': 'Order of transform (4)'
                   }
    affines = None               # Affine transforms common to all transforms

    def __init__(self, **kwargs):
        """Create a transform, optionally loading it from a FITS HDU list.

        Parameters
        ----------
        lms_config : dict, optional
            Instrument configuration (``opticon, pri_ang, ech_ang``).  Default
            is a copy of ``Globals.lms_config_template`` (all ``None``).
        slice_config : dict, optional
            Slice configuration (``slice_no, ech_ord, w_min, w_max``).  Default
            is a copy of ``Globals.slice_config_template``.
        hdu_list : astropy.io.fits.HDUList, optional
            If given, :meth:`ingest_from_hdu_list` is called with ``ext_no``
            and both configurations and the matrices are read from the file,
            overriding ``lms_config`` / ``slice_config``.
        ext_no : int, optional
            Extension index to read when ``hdu_list`` is given.  Required in
            that case (``None`` would index the HDU list and fail).
        matrices : dict, optional
            Pre-computed ``{'a', 'b', 'ai', 'bi'}`` matrices, applied after any
            FITS ingest.

        Notes
        -----
        - Clean-up: the first instantiation in a process silently creates the
          class-level ``Transform.affines`` via :meth:`create_affines`;
          ``lmsdist.run`` relies on this (``_ = Transform()``).
        - Clean-up: the dictionaries passed as ``lms_config`` / ``slice_config``
          are stored by reference, not copied.  ``RayTrace.create_svd_transforms``
          passes the same ``lms_config`` object to every slice transform, so a
          later :meth:`make_hdu_primary` call mutates all of them (adds
          ``n_mats`` / ``mat_ord`` keys).
        """
        self.matrices = {'a': None, 'b': None, 'ai': None, 'bi': None}
        self.lms_configuration = kwargs.get('lms_config', Globals.lms_config_template.copy())
        self.slice_configuration = kwargs.get('slice_config', Globals.slice_config_template.copy())
        if 'hdu_list' in kwargs:
            hdu_list = kwargs.get('hdu_list')
            ext_no = kwargs.get('ext_no')
            self.ingest_from_hdu_list(hdu_list, ext_no)
        if 'matrices' in kwargs:
            self.matrices = kwargs.get('matrices')
        if Transform.affines is None:
            Transform.create_affines()
        return

    @staticmethod
    def create_affines(theta_offsets=[0.]*4,
                       x_mfp_org_offsets=[0.]*4, y_mfp_org_offsets=[0.]*4,
                       x_scale_offsets=[0.]*4, y_scale_offsets=[0.]*4):
        """Create the per-detector affine matrices M (MFP mm -> DFP pixels) and MI.

        For each of the four detectors ``i`` (0-based index, detector number
        ``i + 1``) a 3 x 3 homogeneous matrix is built as ::

            M[i] = [[ sx*cos(t), -sy*sin(t),  sx*x_org ],
                    [ sx*sin(t),  sy*cos(t),  sy*y_org ],
                    [ 0,          0,          1        ]]

        acting on ``(mfp_x, mfp_y, 1)`` to give ``(dfp_x, dfp_y, 1)``, where

        - ``sx = -pix_mm + x_scale_offsets[i]``, ``sy = +pix_mm + y_scale_offsets[i]``
          with ``pix_mm = 1000 / Globals.nom_pix_pitch = 1000 / 18``
          pixels per mm.  The x scale is NEGATIVE, so the column number
          increases towards decreasing ``mfp_x`` (i.e. towards longer
          wavelength).
        - ``t = theta_offsets[i]`` in radians (the nominal rotations are 0).
        - ``x_org = x_mfp_origins[i] + x_mfp_org_offsets[i]`` with
          ``x_mfp_origins = [-xy_fc, +xy_nc, -xy_fc, +xy_nc]`` and
          ``y_org`` likewise with ``y_mfp_origins = [-xy_nc, -xy_nc, +xy_fc, +xy_fc]``,
          where ``xy_fc = Globals.det_size + Globals.det_gap / 2`` (far corner,
          38.364 mm) and ``xy_nc = Globals.det_gap / 2`` (near corner, 1.5 mm).

        With zero rotation this gives, per detector (quadrant convention of
        ``lmsdist_util.Util.mfp_to_dfp``: 1 = x>0,y>0; 2 = x<0,y>0; 3 = x>0,y<0;
        4 = x<0,y<0)::

            det 1: col = pix_mm*(xy_fc - x),  row = pix_mm*(y - xy_nc)
            det 2: col = -pix_mm*(x + xy_nc), row = pix_mm*(y - xy_nc)
            det 3: col = pix_mm*(xy_fc - x),  row = pix_mm*(y + xy_fc)
            det 4: col = -pix_mm*(x + xy_nc), row = pix_mm*(y + xy_fc)

        so column 0 / row 0 is the corner of each detector that gives a
        continuous column count across the gap (det 1 col 2048 meets det 2
        col 0) and rows increasing with ``+mfp_y`` everywhere.  ``MI[i]`` is
        ``numpy.linalg.inv(M[i])``.

        Parameters
        ----------
        theta_offsets : list of float, optional
            Per-detector rotation offsets in radians (default zeros).
        x_mfp_org_offsets, y_mfp_org_offsets : list of float, optional
            Per-detector origin offsets in mm (default zeros).
        x_scale_offsets, y_scale_offsets : list of float, optional
            Per-detector scale offsets in pixels per mm (default zeros).

        Returns
        -------
        dict
            ``{'m': ndarray (4, 3, 3), 'mi': ndarray (4, 3, 3)}``; also stored
            as ``Transform.affines``.

        Notes
        -----
        - Inconsistency: ``Globals.det_gap = 3.0`` mm is used, while
          ``lms_detector.Detector.mosaic_gap = 2.23`` mm and the irdb layout
          imply 2.23 mm.  The origin offsets ``xy_nc`` / ``xy_fc`` and hence
          every pixel coordinate depend on this value.
        - Open question: the translation column ``(sx*x_org, sy*y_org)`` is
          not rotated, so a non-zero ``theta_offsets[i]`` rotates about the
          mosaic origin rather than about the detector's own reference corner.
          ``lmsaiv_opt02`` calls this with measured offsets; whether that is
          the intended pivot should be confirmed.
        - Clean-up: the local ``thetas = [0.]*4`` duplicates the default of
          ``theta_offsets``; mutable list defaults are used for all offsets.
        - Clean-up: ``theta_offsets`` are in radians while the rest of lmsiq
          expresses angles in degrees.
        """
        n_dets = 4
        aff_shape = n_dets, 3, 3
        thetas = [0.]*4
        xy_fc = Globals.det_size + Globals.det_gap / 2.
        xy_nc = Globals.det_gap / 2.
        x_mfp_origins = [-xy_fc, +xy_nc, -xy_fc, +xy_nc]
        y_mfp_origins = [-xy_nc, -xy_nc, +xy_fc, +xy_fc]
        pix_mm = 1000. / Globals.nom_pix_pitch
        y_scales = [pix_mm] * 4
        x_scales = [-pix_mm] * 4
        m, mi = np.zeros(aff_shape), np.zeros(aff_shape)
        for i in range(0, n_dets):
            theta = thetas[i] + theta_offsets[i]
            sx, sy = x_scales[i] + x_scale_offsets[i], y_scales[i] + y_scale_offsets[i]
            x_mfp_org, y_mfp_org = x_mfp_origins[i] + x_mfp_org_offsets[i], y_mfp_origins[i] + y_mfp_org_offsets[i]
            cos_theta, sin_theta = math.cos(theta), math.sin(theta)
            m[i, 0, :] = [sx * cos_theta, -sy * sin_theta, sx * x_mfp_org]
            m[i, 1, :] = [sx * sin_theta, +sy * cos_theta, sy * y_mfp_org]
            m[i, 2, :] = [0., 0., 1.]
            mi[i] = np.linalg.inv(m[i])
        Transform.affines = {'m': m, 'mi': mi}
        return Transform.affines

    def ingest_from_hdu_list(self, hdu_list, ext_no):
        """Load configuration and matrices from one extension of an ``lms_efp_mfp`` FITS file.

        The instrument configuration is read from the primary header using the
        upper-cased keys of ``Globals.lms_config_template`` (``OPTICON``,
        ``PRI_ANG``, ``ECH_ANG``); the slice configuration from extension
        ``ext_no`` using the upper-cased keys of
        ``Globals.slice_config_template`` (``SLICE_NO``, ``ECH_ORD``, ``W_MIN``,
        ``W_MAX``); the matrices via :meth:`read_matrices`.

        Parameters
        ----------
        hdu_list : astropy.io.fits.HDUList
            Open FITS file written by ``Filer.write_svd_transforms``.
        ext_no : int
            1-based extension index of the ``SLICE_<n>`` table to read.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If any expected header keyword is absent (e.g. a file written
            before ``W_MIN`` / ``W_MAX`` were added to the template).

        Notes
        -----
        - Clean-up: ``self.lms_configuration`` is re-created from the template
          here, discarding any ``lms_config`` passed to ``__init__``.
        """
        primary_hdr = hdu_list[0].header
        self.lms_configuration = Globals.lms_config_template.copy()
        for key in self.lms_configuration:          # Get slice 'common' parameters from primary header.
            self.lms_configuration[key] = primary_hdr[key.upper()]
        hdu = hdu_list[ext_no]
        for key in self.slice_configuration:
            self.slice_configuration[key] = hdu.header[key.upper()]
        self.read_matrices(hdu.data)
        return

    def is_match(self, slice_filter):
        """Return True if this transform is for the given slice number and echelle order.

        Parameters
        ----------
        slice_filter : dict
            Must contain ``'slice_no'`` (encoded universal slice number) and
            ``'ech_ord'``.  Any other keys are ignored.

        Returns
        -------
        bool
            ``slice_no`` and ``ech_ord`` both equal.

        Notes
        -----
        - Clean-up: ``lmsdist_util.Util.filter_transform_list`` performs the
          same test generically over arbitrary keys.
        """
        slice_no_match = self.slice_configuration['slice_no'] == slice_filter['slice_no']
        ech_ord_match = self.slice_configuration['ech_ord'] == slice_filter['ech_ord']
        is_match = slice_no_match and ech_ord_match
        return is_match

    def minus(self, tr_in):
        """Return a new transform whose matrices are ``self - tr_in`` term by term.

        Intended for comparing AIT-measured with Zemax-derived transforms
        (``lmsaiv_opt02``).  The new object shares (by reference) this
        transform's ``lms_configuration`` and ``slice_configuration``.

        Parameters
        ----------
        tr_in : Transform
            Transform to subtract; must have all four matrices populated.

        Returns
        -------
        Transform
            Difference transform (``a, b, ai, bi`` each subtracted).

        Notes
        -----
        - Clean-up: not called anywhere in the repository; ``lmsaiv_opt02``
          computes the fractional difference inline instead.
        - Clean-up: configuration dicts are shared, not copied.
        """
        tr_diff = Transform()
        for name in Globals.matrix_names:
            tr_diff.matrices[name] = self.matrices[name] - tr_in.matrices[name]
        tr_diff.lms_configuration = self.lms_configuration
        tr_diff.slice_configuration = self.slice_configuration
        return tr_diff

    def plus(self, tr_in):
        """Return a new transform whose matrices are ``self + tr_in`` term by term.

        Counterpart of :meth:`minus`; e.g. to re-apply a measured difference
        to a design transform.  The new object shares this transform's
        configuration dictionaries by reference.

        Parameters
        ----------
        tr_in : Transform
            Transform to add; must have all four matrices populated.

        Returns
        -------
        Transform
            Sum transform.

        Notes
        -----
        - Clean-up: not called anywhere in the repository.
        - Clean-up: the local variable is named ``tr_diff`` although it holds
          a sum.
        """
        tr_diff = Transform()
        for name in Globals.matrix_names:
            tr_diff.matrices[name] = self.matrices[name] + tr_in.matrices[name]
        tr_diff.lms_configuration = self.lms_configuration
        tr_diff.slice_configuration = self.slice_configuration
        return tr_diff

    def make_hdu_primary(self):
        """Build the primary HDU of an ``lms_efp_mfp`` FITS file from the LMS configuration.

        Header cards are written for every key of ``self.lms_configuration``
        (upper-cased) plus ``N_MATS = 4`` and ``MAT_ORD = 4``, with comments
        from ``Transform.kw_comments``: ``OPTICON``, ``PRI_ANG``, ``ECH_ANG``,
        ``N_MATS``, ``MAT_ORD``.  An empty card with keyword ``W`` (no value,
        no comment) is appended.

        Returns
        -------
        astropy.io.fits.PrimaryHDU
            Header-only primary HDU.

        Raises
        ------
        KeyError
            If ``lms_configuration`` contains a key with no entry in
            ``kw_comments`` (e.g. the ``'wave'`` / ``'ech_order'`` keys added
            by ``PolyFit.wave_to_config``).

        Notes
        -----
        - Open question: the purpose of the empty ``W`` card added in upstream
          commit 9b6143a is not recorded; it looks like an unfinished edit.
        - Inconsistency: the former exclusion of ``slice_no`` / ``spifu_no`` /
          ``ech_ord`` keys was removed; a ``spifu_no`` key would now raise
          KeyError (it has no ``kw_comments`` entry).
        - Clean-up: mutates ``self.lms_configuration`` in place by adding
          ``n_mats`` and ``mat_ord``.  Because ``RayTrace`` shares one dict
          between all slice transforms of a trace, every transform's
          configuration acquires these keys after the first write.
        """
        lms_config = self.lms_configuration            # Get basic configuration parameters
        lms_config['n_mats'] = 4                       # and add useful header information.
        lms_config['mat_ord'] = 4
        cards = []
        for key in lms_config:
            card = Card(key.upper(), lms_config[key], Transform.kw_comments[key])
            cards.append(card)
        cards.append(Card('w'))
        trace_hdr = fits.Header(cards)
        primary_hdu = fits.PrimaryHDU(header=trace_hdr)
        return primary_hdu

    def make_hdu_ext(self):
        """Build the ``SLICE_<n>`` binary-table extension holding this transform.

        Columns ``A``, ``B``, ``AI``, ``BI`` each contain the 16 elements of the
        corresponding 4 x 4 matrix flattened row-major (``ndarray.flatten``),
        FITS format ``'E'`` (32-bit float).  The extension header carries one
        card per key of ``Globals.slice_config_template`` (``SLICE_NO``,
        ``ECH_ORD``, ``W_MIN``, ``W_MAX``) with comments from ``kw_comments``,
        and ``EXTNAME = 'SLICE_{slice_no}'`` where ``slice_no`` is the encoded
        universal slice number (e.g. ``SLICE_13`` nominal, ``SLICE_312``
        extended for pslice 3 / fslice 12).

        Returns
        -------
        astropy.io.fits.BinTableHDU

        Raises
        ------
        KeyError
            If ``slice_configuration`` lacks ``w_min`` / ``w_max``, i.e. if
            ``RayTrace.find_wavelength_bounds`` was not run before writing.
        AttributeError
            If any matrix is still ``None``.

        Notes
        -----
        - Clean-up: format ``'E'`` (float32) truncates the fitted double-
          precision coefficients; ``'D'`` should be used (see module notes).
        - Inconsistency: the matrix order is hard-wired to 4 x 4 here through
          the column length while ``Globals.svd_order`` is consulted by
          :meth:`read_matrices`; the two must agree.
        """
        mats = self.matrices
        a, b, ai, bi = mats['a'], mats['b'], mats['ai'], mats['bi']
        col_a = fits.Column(name='A', array=a.flatten(), format='E')
        col_b = fits.Column(name='B', array=b.flatten(), format='E')
        col_ai = fits.Column(name='AI', array=ai.flatten(), format='E')
        col_bi = fits.Column(name='BI', array=bi.flatten(), format='E')

        keys = Globals.slice_config_template.keys()
        sli_config = self.slice_configuration
        cards = []
        for key in keys:
            val = sli_config[key]
            card = Card(key.upper(), val, self.kw_comments[key])
            cards.append(card)
        hdr = fits.Header(cards)
        hdu_name = "SLICE_{:d}".format(sli_config['slice_no'])
        bintable_hdu = fits.BinTableHDU.from_columns([col_a, col_b, col_ai, col_bi],
                                                     header=hdr, name=hdu_name)
        return bintable_hdu

    def read_matrices(self, table):
        """Populate ``self.matrices`` from a ``SLICE_<n>`` table's A, B, AI, BI columns.

        Each column (16 values) is reshaped row-major to
        ``(Globals.svd_order, Globals.svd_order)`` = (4, 4), inverting the
        ``flatten`` in :meth:`make_hdu_ext`.

        Parameters
        ----------
        table : astropy.io.fits.FITS_rec
            The ``.data`` of a binary-table HDU with columns named
            ``'A', 'B', 'AI', 'BI'``.

        Returns
        -------
        None

        Notes
        -----
        - Clean-up: the resulting arrays keep the file's float32 dtype
          (big-endian ``>f4``), so all downstream arithmetic with these
          matrices is single precision unless the caller casts.
        """
        mat_order = Globals.svd_order
        mat_shape = mat_order, mat_order
        for mat_name in self.matrices:
            col_data = table[mat_name.upper()]
            matrix = np.reshape(col_data, mat_shape)
            self.matrices[mat_name] = matrix
        return

    def _make_fits_tag(self, val):
        """Return a file-name-safe tag for a float (``'-'`` -> ``'m'``, ``'.'`` -> ``'p'``).

        Parameters
        ----------
        val : float
            Value formatted as ``'{:5.3f}'``.

        Returns
        -------
        str
            E.g. ``-1.5`` -> ``'m1p500'``.

        Notes
        -----
        - Clean-up: not called anywhere; file-name tags are produced by
          ``Globals.make_lms_config_id`` instead.
        """
        val_str = "{:5.3f}".format(val)
        tag = val_str.replace('-', 'm')
        tag = tag.replace('.','p')
        return tag

    @staticmethod
    def read_pickle(pickle_path):
        """Unpickle and return the object stored at ``pickle_path`` (``.pkl`` appended if missing).

        Parameters
        ----------
        pickle_path : str
            Path with or without the ``.pkl`` extension.

        Returns
        -------
        object

        Notes
        -----
        - Clean-up: duplicate of ``lms_filer.Filer.read_pickle``; unused.
        """
        if pickle_path[-4:] != '.pkl':
            pickle_path += '.pkl'
        file = open(pickle_path, 'rb')
        python_object = pickle.load(file)
        file.close()
        return python_object

    @staticmethod
    def write_pickle(pickle_path, python_object):
        """Pickle ``python_object`` to ``pickle_path + '.pkl'``.

        Parameters
        ----------
        pickle_path : str
            Path WITHOUT extension; ``.pkl`` is always appended (unlike
            :meth:`read_pickle`, which tolerates an existing extension).
        python_object : object

        Returns
        -------
        None

        Notes
        -----
        - Clean-up: duplicate of ``lms_filer.Filer.write_pickle``; unused.
        """
        file = open(pickle_path + '.pkl', 'wb')
        pickle.dump(python_object, file)
        file.close()
        return

    def _get_results_path(self, data_id, data_type):
        """Build an ``lmsiq`` results CSV path.  Dead code copied from ``lms_filer.Filer``.

        Parameters
        ----------
        data_id : sequence
            Seven-element ``(dataset, slice_subfolder, ipc_tag, process_level,
            config_folder, mcrun_tag, axis)``.
        data_type : str
            One of the keys of the internal ``type_tags`` table
            (``'xcentroids'``, ``'photometry'``, ``'ee_spectral'`` ...).

        Returns
        -------
        str
            Would be the results path.

        Raises
        ------
        AttributeError
            Always: ``self.output_folder`` and ``self.get_folder`` do not exist
            on ``Transform``.

        Notes
        -----
        - Bug: references attributes of ``Filer`` on a ``Transform`` instance.
        - Clean-up: verbatim copy of ``lms_filer.Filer._get_results_path``,
          which belongs to the ``lmsiq`` image-quality program, not to
          ``lmsdist``.  Remove.
        """
        dataset, slice_subfolder, ipc_tag, process_level, config_folder, mcrun_tag, axis = data_id
        folder = self.output_folder + slice_subfolder
        folder += ipc_tag + '/' + process_level + '/' + data_type
        folder = self.get_folder(folder)

        type_tags = {'xcentroids': '_xcen', 'ycentroids': '_ycen',
                     'xfwhm_gau': '_xfwhm', 'photometry': '_phot',
                     'ee_spectral': '_eex', 'ee_spatial': '_eey',
                     'lsf_spectral': '_lsx', 'lsf_spatial': '_lsy',
                     'ee_dfp_spectral': '_eex_dfp', 'ee_dfp_spatial': '_eey_dfp',
                     'lsf_dfp_spectral': '_lsx_dfp', 'lsf_dfp_spatial': '_lsy_dfp',
                     }
        type_tag = type_tags[data_type]
        config_tag = config_folder[0:-1]
        slice_tag = slice_subfolder[:-1] + '_'
        file_name = slice_tag + ipc_tag + '_' + process_level + type_tag + '_wav_' + config_tag + '.csv'
        path = folder + file_name
        return path
