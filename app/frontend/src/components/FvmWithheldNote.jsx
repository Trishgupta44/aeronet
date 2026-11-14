// The one sentence every page uses when the finite-volume (FVM) Cp curves are not served (public deployment, see docs/DEPLOY.md).
export default function FvmWithheldNote({ children }) {
  return (
    <p className="hint fvm-note" role="note">
      FVM reference curves withheld: NNfoil has no licence for redistribution; error numbers computed against them are shown. {children}
    </p>
  );
}
