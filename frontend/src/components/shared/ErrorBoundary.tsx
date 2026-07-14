import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[React Error Boundary]: Catching error:', error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.href = '/';
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex flex-col items-center justify-center bg-black px-6 text-center select-none relative">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(229,9,20,0.08),transparent_70%)] pointer-events-none" />

          <div className="relative z-10 max-w-md w-full flex flex-col items-center gap-6">
            <div className="w-20 h-20 rounded-full flex items-center justify-center border-2 border-primary/50 border-dashed animate-spin [animation-duration:12s]">
              <span className="text-primary text-3xl font-extrabold">!</span>
            </div>

            <div className="space-y-2">
              <h1 className="text-3xl font-extrabold uppercase tracking-wide text-white">
                Signal Interrupted
              </h1>
              <p className="text-muted-foreground text-sm leading-relaxed">
                The cinematic feed was cut due to a projection error. We are currently calibrating
                our reels.
              </p>
              {this.state.error && (
                <div className="mt-4 p-4 bg-muted border border-border text-xs text-primary/70 rounded-md overflow-x-auto text-left max-h-36 max-w-full font-mono">
                  {this.state.error.stack || this.state.error.toString()}
                </div>
              )}
            </div>

            <button
              onClick={this.handleReset}
              className="px-6 py-2.5 bg-primary text-white font-semibold rounded-md shadow-lg shadow-primary/25 hover:bg-primary/95 transition-all duration-300 transform hover:scale-105 active:scale-95 text-sm uppercase tracking-wider cursor-pointer"
            >
              Restart Feed
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
