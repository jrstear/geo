import React, { Component } from "React";
import ReactDOM from "ReactDOM";

export default class TaskView extends Component {
    state = {
        portalContainer: null,
        showModal: false,
        loading: false,
        error: null,
        result: null,
        useReconstruction: true,
        csvFile: null,
    };

    sentinelRef = React.createRef();

    _findAndPortal() {
        // Walk up the DOM to the nearest .expanded-panel, then inject our
        // button into the existing .action-buttons row so it sits alongside
        // Download / Map / 3D Model / Report.
        const sentinel = this.sentinelRef.current;
        if (!sentinel) return false;
        let el = sentinel.parentElement;
        while (el && !el.classList.contains("expanded-panel")) {
            el = el.parentElement;
        }
        if (!el) return false;
        const actionButtonsDiv = el.querySelector(".action-buttons");
        if (actionButtonsDiv) {
            this.setState({ portalContainer: actionButtonsDiv });
            return true;
        }
        return false;
    }

    componentDidMount() {
        // Try immediately; if .action-buttons isn't in the DOM yet (sibling
        // component still mounting) retry after the current render cycle.
        if (!this._findAndPortal()) {
            setTimeout(() => this._findAndPortal(), 0);
        }
    }

    openModal  = () => this.setState({ showModal: true,  error: null, result: null });
    closeModal = () => this.setState({ showModal: false, loading: false });

    onCsvChange   = (e) => this.setState({ csvFile: e.target.files[0] || null });
    onReconChange = (e) => this.setState({ useReconstruction: e.target.checked });

    onGenerate = async () => {
        const { csvFile, useReconstruction } = this.state;
        const { task, apiURL } = this.props;

        if (!csvFile) {
            this.setState({ error: "Please select an Emlid CSV file." });
            return;
        }

        this.setState({ loading: true, error: null, result: null });

        const formData = new FormData();
        formData.append("emlid_csv", csvFile);
        formData.append("use_reconstruction", useReconstruction ? "true" : "false");

        try {
            const resp = await fetch(`${apiURL}/task/${task.id}/generate`, {
                method: "POST",
                body: formData,
                headers: { "X-CSRFToken": this.getCookie("csrftoken") },
                credentials: "same-origin",
            });
            const data = await resp.json();
            if (!resp.ok) {
                this.setState({ error: data.error || "Server error", loading: false });
            } else {
                this.setState({ result: data, loading: false });
            }
        } catch (e) {
            this.setState({ error: e.message, loading: false });
        }
    };

    getCookie(name) {
        const v = document.cookie.match("(^|;) ?" + name + "=([^;]*)(;|$)");
        return v ? v[2] : null;
    }

    render() {
        const { portalContainer, showModal, loading, error, result, useReconstruction } = this.state;

        const button = (
            <div style={{ display: "inline-block", marginLeft: "4px" }}>
                <button
                    className="btn btn-sm btn-primary"
                    onClick={this.openModal}
                    title="Sight GCPs from Emlid CSV"
                >
                    <i className="fa fa-crosshairs fa-fw" />
                    <span className="hidden-xs hidden-sm"> GCPSighter</span>
                </button>

                {showModal && (
                    <div className="modal" style={{ display: "block", background: "rgba(0,0,0,0.5)" }}>
                        <div className="modal-dialog">
                            <div className="modal-content">
                                <div className="modal-header">
                                    <h5 className="modal-title">GCPSighter</h5>
                                    <button type="button" className="close" onClick={this.closeModal}>
                                        <span>&times;</span>
                                    </button>
                                </div>
                                <div className="modal-body">
                                    <div className="form-group">
                                        <label>Emlid CSV</label>
                                        <input
                                            type="file"
                                            className="form-control-file"
                                            accept=".csv"
                                            onChange={this.onCsvChange}
                                        />
                                    </div>
                                    <div className="form-check mt-2">
                                        <input
                                            type="checkbox"
                                            className="form-check-input"
                                            id="use-recon"
                                            checked={useReconstruction}
                                            onChange={this.onReconChange}
                                        />
                                        <label className="form-check-label" htmlFor="use-recon">
                                            Use reconstruction.json if available (more accurate)
                                        </label>
                                    </div>
                                    {error && <div className="alert alert-danger mt-3">{error}</div>}
                                    {result && (
                                        <div className="alert alert-success mt-3">
                                            <p>GCP estimates generated successfully.</p>
                                            <a href={result.gcp_list_txt} className="btn btn-sm btn-primary" download>
                                                Download gcp_list.txt
                                            </a>
                                        </div>
                                    )}
                                </div>
                                <div className="modal-footer">
                                    <button className="btn btn-secondary" onClick={this.closeModal} disabled={loading}>
                                        Close
                                    </button>
                                    <button className="btn btn-primary" onClick={this.onGenerate} disabled={loading}>
                                        {loading ? (
                                            <React.Fragment>
                                                <span className="spinner-border spinner-border-sm mr-1" role="status" />
                                                Running…
                                            </React.Fragment>
                                        ) : "Generate"}
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        );

        return (
            <React.Fragment>
                {/* Sentinel stays in the plugin-action-buttons row (hidden) */}
                <div ref={this.sentinelRef} style={{ display: "none" }} />
                {portalContainer
                    ? ReactDOM.createPortal(button, portalContainer)
                    : button}
            </React.Fragment>
        );
    }
}
