"""Data access layer for reading TensorBoard data via DataProvider."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from tensorboard.context import RequestContext
from tensorboard.data.provider import RunTagFilter

from tensorboard_ai.constants import MAX_SCALAR_POINTS


class TBDataAccess:
    def __init__(self, data_provider: Any, experiment_id: str = ""):
        self._provider = data_provider
        self._experiment_id = experiment_id
        self._ctx = RequestContext()

    def list_runs_and_tags(self) -> Dict[str, Dict[str, List[str]]]:
        """Return {run: {scalars: [tag], tensors: [tag], blob_sequences: [tag]}}."""
        result: Dict[str, Dict[str, List[str]]] = {}

        scalars = self._provider.list_scalars(
            ctx=self._ctx,
            experiment_id=self._experiment_id,
            plugin_name="scalars",
        )
        for run, tag_to_content in scalars.items():
            result.setdefault(run, {"scalars": [], "tensors": [], "blob_sequences": []})
            result[run]["scalars"] = sorted(tag_to_content.keys())

        try:
            tensors = self._provider.list_tensors(
                ctx=self._ctx,
                experiment_id=self._experiment_id,
                plugin_name="histograms",
            )
            for run, tag_to_content in tensors.items():
                result.setdefault(run, {"scalars": [], "tensors": [], "blob_sequences": []})
                result[run]["tensors"] = sorted(tag_to_content.keys())
        except Exception:
            pass

        try:
            blob_sequences = self._provider.list_blob_sequences(
                ctx=self._ctx,
                experiment_id=self._experiment_id,
                plugin_name="images",
            )
            for run, tag_to_content in blob_sequences.items():
                result.setdefault(run, {"scalars": [], "tensors": [], "blob_sequences": []})
                result[run]["blob_sequences"] = sorted(tag_to_content.keys())
        except Exception:
            pass

        return result

    def read_scalar_summary(
        self,
        run: str,
        tag: str,
        downsample: int = MAX_SCALAR_POINTS,
    ) -> Dict[str, Any]:
        """Read scalar data and return a summary dict."""
        rtf = RunTagFilter(runs=[run], tags=[tag])
        data = self._provider.read_scalars(
            ctx=self._ctx,
            experiment_id=self._experiment_id,
            plugin_name="scalars",
            run_tag_filter=rtf,
            downsample=downsample,
        )
        points = data.get(run, {}).get(tag, [])
        if not points:
            return {"run": run, "tag": tag, "count": 0}

        values = [p.value for p in points]
        steps = [p.step for p in points]
        return {
            "run": run,
            "tag": tag,
            "count": len(points),
            "min": min(values),
            "max": max(values),
            "first": values[0],
            "last": values[-1],
            "step_range": [steps[0], steps[-1]],
            "sampled_points": [{"step": p.step, "value": p.value} for p in points],
        }

    def read_scalars_for_tags(
        self,
        run_tag_pairs: List[Tuple[str, str]],
        downsample: int = MAX_SCALAR_POINTS,
    ) -> Dict[str, Dict[str, list]]:
        """Read multiple scalar series at once."""
        runs = set()
        tags = set()
        for run, tag in run_tag_pairs:
            runs.add(run)
            tags.add(tag)

        rtf = RunTagFilter(runs=list(runs), tags=list(tags))
        data = self._provider.read_scalars(
            ctx=self._ctx,
            experiment_id=self._experiment_id,
            plugin_name="scalars",
            run_tag_filter=rtf,
            downsample=downsample,
        )

        # Filter to only the requested (run, tag) pairs
        requested = set(run_tag_pairs)
        result: Dict[str, Dict[str, list]] = {}
        for run, tag_data in data.items():
            for tag, points in tag_data.items():
                if (run, tag) in requested:
                    result.setdefault(run, {})[tag] = [
                        {"step": p.step, "value": p.value, "wall_time": p.wall_time}
                        for p in points
                    ]
        return result

    def read_hyperparameters(self) -> Dict[str, Dict[str, Any]]:
        """Read hyperparameters per run. Returns {run: {hparam_name: value}}.

        Tries the hparams API first, falls back to text plugin tensors.
        """
        result: Dict[str, Dict[str, Any]] = {}

        # Try hparams API
        try:
            hp_result = self._provider.list_hyperparameters(
                ctx=self._ctx,
                experiment_ids=[self._experiment_id],
            )
            for sg in hp_result.session_groups:
                run_name = sg.root.run or sg.root.experiment_id
                if sg.sessions:
                    run_name = sg.sessions[0].run or run_name
                hparams: Dict[str, Any] = {}
                for hv in sg.hyperparameter_values:
                    if hv.value is not None:
                        hparams[hv.hyperparameter_name] = hv.value
                if hparams:
                    result[run_name] = hparams
        except Exception:
            pass

        if result:
            return result

        # Fallback: read text plugin tensors (some frameworks store hparams as text)
        try:
            text_tags = self._provider.list_tensors(
                ctx=self._ctx,
                experiment_id=self._experiment_id,
                plugin_name="text",
            )
            for run, tag_to_content in text_tags.items():
                for tag in tag_to_content:
                    rtf = RunTagFilter(runs=[run], tags=[tag])
                    data = self._provider.read_tensors(
                        ctx=self._ctx,
                        experiment_id=self._experiment_id,
                        plugin_name="text",
                        run_tag_filter=rtf,
                        downsample=10,
                    )
                    points = data.get(run, {}).get(tag, [])
                    if points:
                        val = points[-1].numpy
                        if hasattr(val, 'item'):
                            val = val.item()
                        if isinstance(val, bytes):
                            val = val.decode("utf-8", errors="replace")
                        result.setdefault(run, {})[tag] = str(val)
        except Exception:
            pass

        return result

    def get_hyperparameters_context(self) -> str:
        """Build text summary of hyperparameters that differ across runs.

        Only includes hparams where at least two runs have different values.
        Returns empty string if no differing hparams found.
        """
        hparams_by_run = self.read_hyperparameters()
        if not hparams_by_run:
            return ""

        runs = sorted(hparams_by_run.keys())
        if len(runs) < 2:
            # Single run: include all hparams (no filtering needed)
            run = runs[0]
            lines = ["Hyperparameters:"]
            for name, val in sorted(hparams_by_run[run].items()):
                lines.append(f"  {name}: {val}")
            return "\n".join(lines) if len(lines) > 1 else ""

        # Collect all hparam names
        all_names: set = set()
        for hparams in hparams_by_run.values():
            all_names.update(hparams.keys())

        # Find hparams that differ across runs
        differing: Dict[str, Dict[str, Any]] = {}  # {hparam_name: {run: value}}
        common: Dict[str, Any] = {}  # {hparam_name: value}

        for name in sorted(all_names):
            values_by_run: Dict[str, Any] = {}
            for run in runs:
                if name in hparams_by_run.get(run, {}):
                    values_by_run[run] = hparams_by_run[run][name]

            unique_vals = set(str(v) for v in values_by_run.values())
            if len(unique_vals) > 1:
                differing[name] = values_by_run
            elif unique_vals:
                common[name] = next(iter(values_by_run.values()))

        if not differing and not common:
            return ""

        lines = ["Hyperparameters:"]

        if differing:
            lines.append("  Differing across runs:")
            for name, run_vals in sorted(differing.items()):
                vals_str = ", ".join(f"{r}={v}" for r, v in sorted(run_vals.items()))
                lines.append(f"    {name}: {vals_str}")

        if common:
            lines.append("  Common across all runs:")
            for name, val in sorted(common.items()):
                lines.append(f"    {name}: {val}")

        return "\n".join(lines)

    def get_data_context(self) -> str:
        """Build a text summary of available data for system prompt."""
        runs_and_tags = self.list_runs_and_tags()
        if not runs_and_tags:
            return "No training data available."

        lines = ["Available training data:"]
        for run, tags in sorted(runs_and_tags.items()):
            lines.append(f"\nRun: {run}")
            if tags["scalars"]:
                lines.append(f"  Scalars: {', '.join(tags['scalars'])}")
            if tags["tensors"]:
                lines.append(f"  Tensors: {', '.join(tags['tensors'])}")
            if tags["blob_sequences"]:
                lines.append(f"  Blob sequences: {', '.join(tags['blob_sequences'])}")

        return "\n".join(lines)
