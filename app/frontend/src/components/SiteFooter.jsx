import Glyph from './Glyph.jsx';

export default function SiteFooter() {
  return (
    <footer className="site-footer" id="credit">
      <div className="container footer-in">
        <div>
          <a className="brand" href="#home"><Glyph /> <span>AeroNet</span></a>
          <p className="footer-lead">An airfoil flow explorer for screening, not a certification tool.</p>
        </div>
        <div className="footer-notes">
          <p>
            Flow solver for this view: Hess–Smith panel method (inviscid, subsonic, Prandtl–Glauert correction). The neural solver is the NNfoil
            PINN by Cao, Song and Zhang (arXiv 2401.08705), used with credit and with the author's permission (its repository has no licence file; see docs/DEPLOY.md).
          </p>
          <p>
            Scope: inviscid, subsonic, 2D. The drag shown is a numerical error, not a prediction. It complements high-fidelity CFD and does not replace it.
          </p>
          <p><a href="#results">Read the results</a></p>
        </div>
      </div>
    </footer>
  );
}
