/**
 * Mock implementations of external dependencies to make the neuromorphic modules work
 * without requiring external libraries.
 */

// Mock nanoid
export function nanoid(): string {
  return Math.random().toString(36).substring(2) + Date.now().toString(36);
}

// Mock Subject from rxjs
export class Subject<T> {
  private observers: Array<(value: T) => void> = [];

  next(value: T): void {
    this.observers.forEach(observer => observer(value));
  }

  subscribe(callback: (value: T) => void): { unsubscribe: () => void } {
    this.observers.push(callback);
    return {
      unsubscribe: () => {
        const index = this.observers.indexOf(callback);
        if (index !== -1) {
          this.observers.splice(index, 1);
        }
      }
    };
  }
}

// Mock BehaviorSubject from rxjs
export class BehaviorSubject<T> extends Subject<T> {
  private _value: T;

  constructor(initialValue: T) {
    super();
    this._value = initialValue;
  }

  get value(): T {
    return this._value;
  }

  next(value: T): void {
    this._value = value;
    super.next(value);
  }

  subscribe(callback: (value: T) => void): { unsubscribe: () => void } {
    callback(this._value);
    return super.subscribe(callback);
  }
}

// Simple LSTM mock for brain.js
export class LSTMTimeStep {
  private options: any;
  private model: any = {};
  private trained: boolean = false;

  constructor(options: any) {
    this.options = options;
  }

  train(data: any[], options: any = {}): any {
    this.trained = true;
    return { error: 0.05, iterations: 1000 };
  }

  run(input: any[]): any[] {
    // Return dummy predictions
    return input.map(i => i * 1.1);
  }

  forecast(input: any[], count: number): any[] {
    // Return dummy forecast
    const result = [];
    let last = input[input.length - 1];
    for (let i = 0; i < count; i++) {
      last = last * 0.95 + Math.random() * 0.1;
      result.push(last);
    }
    return result;
  }
}

// Mock brain.js namespace
export const brain = {
  LSTMTimeStep,
  recurrent: {
    LSTMTimeStep
  }
};

// Mock neataptic namespace
export const neataptic = {
  Architect: {
    Perceptron: function(a: number, b: number, c: number) {
      return {
        activate: (input: number[]) => input.map(i => Math.tanh(i)),
        propagate: (rate: number, target: number[]) => {},
        toJSON: () => ({}),
        fromJSON: (json: any) => {},
      };
    }
  },
  methods: {
    activation: {
      TANH: 'TANH'
    }
  }
};

// Mock pcap for browser environments
export const pcap = {
  createSession: () => {
    return {
      on: (_event: string, _callback: any) => {},
      close: () => {}
    };
  },
  Packet: {
    prototype: {
      decode: () => ({})
    }
  }
};
