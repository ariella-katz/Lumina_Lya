"""Tiny end-to-end checks; CUDA runs only with an explicit --cuda argument."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cpu"))
sys.path.insert(0, str(ROOT / "cpu" / "tests"))
import reference as ref
from test_reference import fixture

EXECUTABLE = str(Path(sys.argv.pop(1)).resolve())
BACKEND = "cuda" if "--cuda" in sys.argv else "cpu"
if "--cuda" in sys.argv:
    sys.argv.remove("--cuda")


class Pipeline(unittest.TestCase):
    def setUp(self):
        artifacts = ROOT / "gpu" / "test-artifacts"
        artifacts.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=artifacts)
        self.root = Path(self.temp.name)
        self.input = self.root / "input.h5"
        self.sources = self.root / "sources.txt"
        fixture(self.input)
        self.sources.write_text("6.0025,0.1\n6.0\n")

    def tearDown(self):
        self.temp.cleanup()

    def command(self, *args, success=True):
        result = subprocess.run([EXECUTABLE, *map(str, args)], text=True,
                                capture_output=True, timeout=60)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def run_native(self, output, *extra, success=True):
        return self.command("run", "--input", self.input, "--z0-file", self.sources,
                            "--output-dir", output, "--backend", BACKEND,
                            "--dv-step", "1000", "--tile-size", "2", "--depth-slab", "2",
                            "--products", "maps,spectra,cumulative", *extra, success=success)

    @staticmethod
    def products(directory):
        result = {}
        for path in directory.rglob("*.hdf5"):
            with h5py.File(path) as f:
                result[float(f.attrs["SourceRedshiftRequested"])] = path
        return result

    def compare(self, a, b, *, native_reference=False):
        with h5py.File(a) as aa, h5py.File(b) as bb:
            for key in ("taus", "tau_band_avgs", "T_bands", "T_cum_bands", "T_redshifts"):
                np.testing.assert_allclose(aa[key][:], bb[key][:], rtol=1e-8, atol=1e-8,
                                           err_msg=key)
            self.assertTrue(np.all(aa["taus"][:] >= 0))
            if aa["T_cum_bands"].shape[1]:
                np.testing.assert_array_equal(aa["T_bands"][:], aa["T_cum_bands"][:, -1])
            else:
                np.testing.assert_array_equal(aa["T_bands"][:], np.ones(5))

    def test_modes_against_independent_reference(self):
        for sampling in ("band", "comb"):
            for profile in ("voigt", "delta"):
                name = sampling + profile
                out = self.root / name
                self.run_native(out, "--sampling", sampling, "--profile", profile)
                config = ref.Config(sampling=sampling, profile=profile, dv_step=1000,
                                    tile_size=2, depth_slab=2,
                                    products=("maps", "spectra", "cumulative"))
                references = ref.run(self.input, [[6.0025, .1], 6.], self.root / (name + "ref"), config)
                actual = self.products(out)
                self.assertEqual(len(actual), 2)
                for reference in references:
                    with h5py.File(reference) as f:
                        source = float(f.attrs["SourceRedshiftRequested"])
                    self.compare(actual[source], reference, native_reference=True)

    def test_legacy_and_slab_invariance(self):
        self.sources.write_text("6.0025\n")
        self.run_native(self.root / "legacy", "--legacy")
        expected = ref.run(self.input, [6.0025], self.root / "legacyref",
                           ref.Config(legacy=True, tile_size=2, depth_slab=2,
                                      products=("maps", "spectra", "cumulative")))[0]
        self.compare(next(iter(self.products(self.root / "legacy").values())), expected)
        self.run_native(self.root / "a")
        self.run_native(self.root / "b", "--tile-size", "3", "--depth-slab", "1")
        self.compare(next(iter(self.products(self.root / "a").values())),
                     next(iter(self.products(self.root / "b").values())))
        # The GPU has separate final-spectrum and cumulative kernels.
        self.command("run", "--input", self.input, "--z0-file", self.sources,
                     "--output-dir", self.root / "normal", "--backend", BACKEND,
                     "--dv-step", "1000", "--products", "maps,spectra")
        normal = next(iter(self.products(self.root / "normal").values()))
        with h5py.File(normal) as a, h5py.File(next(iter(self.products(self.root / "a").values()))) as b:
            np.testing.assert_allclose(a["taus"][:], b["taus"][:], rtol=1e-8, atol=1e-8)

    def test_merge_unequal_regions_and_resume(self):
        self.sources.write_text("6.0025\n")
        self.run_native(self.root / "whole")
        self.run_native(self.root / "parts", "--region", "0,1,0,3")
        self.run_native(self.root / "parts", "--region", "1,3,0,3")
        parts = sorted((self.root / "parts").rglob("*.hdf5"))
        merged = self.root / "merged.h5"
        self.command("merge", "--output", merged, *parts)
        self.compare(merged, next(iter(self.products(self.root / "whole").values())))
        self.command("merge", "--output", merged, "--resume", *parts)
        self.command("merge", "--output", self.root / "bad.h5", parts[0], parts[0], success=False)
        self.run_native(self.root / "whole", "--resume")
        self.run_native(self.root / "whole", "--resume", "--dv-step", "500", success=False)
        self.run_native(self.root / "whole", success=False)

    def test_normal_frequency_tiles_multiple_sources(self):
        # More than 128 channels exercises multiple shared-memory frequency
        # tiles; both sources traverse gas rather than having an empty path.
        self.sources.write_text("6.0025\n6.0015\n")
        for backend, name, tile in ((BACKEND, "normal-many", "2"), ("cpu", "normal-reference", "3")):
            self.command("run", "--input", self.input, "--z0-file", self.sources,
                         "--output-dir", self.root / name, "--backend", backend,
                         "--dv-step", "20", "--tile-size", tile,
                         "--depth-slab", "1", "--products", "maps,spectra")
        actual = self.products(self.root / "normal-many")
        expected = self.products(self.root / "normal-reference")
        for source in actual:
            with h5py.File(actual[source]) as a, h5py.File(expected[source]) as b:
                self.assertGreater(a['taus'].shape[-1], 128)
                for key in ('taus', 'tau_band_avgs', 'T_bands'):
                    np.testing.assert_allclose(a[key][:], b[key][:], rtol=1e-8, atol=1e-8)

    def test_metadata_and_input_failures(self):
        self.command("inspect", "--input", self.input)
        self.run_native(self.root / "dry", "--dry-run")
        self.assertFalse((self.root / "dry").exists())
        self.sources.write_text("5.99\n")
        self.run_native(self.root / "badsource", success=False)
        self.sources.write_text("6.003\n")
        self.run_native(self.root / "lowmem", "--host-memory-mib", "0.001", success=False)
        self.command("inspect", "--input", self.input, "--dv-step", ".0001",
                     "--host-memory-mib", "1", success=False)
        with h5py.File(self.input, "r+") as f:
            f["Header"].attrs.modify("HubbleParam", .6774)
            del f["Header"].attrs["HubbleParam"]
            f["Header"].attrs["HubbleParam"] = [.6774, .6774]
        self.command("inspect", "--input", self.input, success=False)

    def test_unused_upstream_cells_are_not_read(self):
        self.sources.write_text("6.0015\n")
        with h5py.File(self.input, 'r+') as f:
            f['Temperature'][:, :, 0] = np.nan
        self.run_native(self.root / 'clipped')
        reference = ref.run(self.input, [6.0015], self.root / 'clipped-reference',
                            ref.Config(dv_step=1000, products=('maps', 'spectra', 'cumulative')))[0]
        self.compare(next(iter(self.products(self.root / 'clipped').values())), reference)

    def test_virtual_and_float64(self):
        source = self.root / "density.h5"
        fixture(self.input, transparent=True, dtype="f8")
        with h5py.File(self.input, "r+") as f, h5py.File(source, "w") as sf:
            sf["Density"] = f["Density"][:]
            del f["Density"]
            layout = h5py.VirtualLayout((3, 3, 3), dtype="f8")
            layout[:] = h5py.VirtualSource(source.name, "Density", shape=(3, 3, 3))
            f.create_virtual_dataset("Density", layout)
        self.run_native(self.root / "vds")
        for path in self.products(self.root / "vds").values():
            with h5py.File(path) as f:
                np.testing.assert_array_equal(f["taus"][:], 0)
                np.testing.assert_allclose(f["T_bands"][:], 1, atol=1e-15)
        source.unlink()
        self.command("inspect", "--input", self.input, success=False)


if __name__ == "__main__":
    unittest.main()
