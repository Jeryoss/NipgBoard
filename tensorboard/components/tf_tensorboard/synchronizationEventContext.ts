namespace tf_tensorboard {

    export type SelectionChangedListener =
        (selectedPointIndices: number[]) => void;
    
    
    export interface SelectionChangedListeners {
        [index: string]: SelectionChangedListener;
    }
    
    export type HoverChangedListener =
        (filename: string) => void;
    
    export interface HoverChangedListeners {
        [index: string]: HoverChangedListener;
    }

    export type HelpChangedListener =
        (text: string) => void;

    export interface HelpChangedListeners {
        [index: string]: HelpChangedListener;
    }

    
    export let currentSession;
    
    export function changeSelection(session: string) {
      currentSession = session;
    }

    export let logged = false;

    export let notifications = [];

    export let viewmode = false;

    export let boardPath = "";

    export let viewlink;

    export let sublogdir = '';
    export let username = '';
    export let password = '';

    export let disablehighres = false;
    export let disablesprite = false;
    export let disablepairwise = false;
    export let disabletraining = false;
    export let disabledownload = false;
    export let disableexport = false;
    export let modifylabels = false;
    export let unifyandoutlier = false;
    export let disableclustering = false;



    
    export let selectedRun;

    export let selectionChangedListeners: SelectionChangedListeners = {};
    
    export let hoverChangedListeners: HoverChangedListeners = {};

    export let helpChangedListeners: HelpChangedListeners = {};
    
    /** Registers a callback to be invoked when the selection changes. */
    export function registerSelectionChangedListener(listener: SelectionChangedListener, name: string) {
      selectionChangedListeners[name] = listener;
    }
    /**
     * Notify the selection system that a client has changed the selected point
     * set.
     */
    export function notifySelectionChanged(newSelectedPointIndices: number[], multiSelect: boolean, name: string) {
      for(var sl in selectionChangedListeners) {
        if (sl !== name) {
          selectionChangedListeners[sl](newSelectedPointIndices);
        }
      }
    }

    export function registerHelpChangedListener(listener: HelpChangedListener, name: string) {
        helpChangedListeners[name] = listener;
    }

    export function notifyHelpChanged(text: string, name: string) {
        for (var hl in helpChangedListeners) {
          if (hl !== name) {
            helpChangedListeners[hl](text);
          }
        }
    }
    
    export function registerHoverChangedListener(listener: HoverChangedListener, name: string) {
      hoverChangedListeners[name] = listener;
    }

    export function handleAddNewNotification(obj){
      let multidash = document.querySelector("vz-multidash");
      (multidash as any).handleAddNewNotification(obj);
    }
    }
    