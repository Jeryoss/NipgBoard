namespace vz_modelmanager {

export type Spec = {
  is: string; properties?: {
    [key: string]:
        (Function |
         {
           type: Function, value?: any;
           readonly?: boolean;
           notify?: boolean;
           observer?: string;
         })
  };
  observers?: string[];
};

export function PolymerElement(spec: Spec) {
  return Polymer.Class(spec as any) as{new (): PolymerHTMLElement};
}

export interface PolymerHTMLElement extends HTMLElement, polymer.Base {}

} 
