# Pla de Refactorització del Constructor de Models PyTorch

**Objectiu:** Modificar el constructor de models existent per suportar arquitectures de xarxes neuronals no només seqüencials, sinó també aquelles amb connexions complexes entre capes (ex: skip connections, múltiples entrades per a una capa).

**Tasques a completar:**

- [ ] **Definició d'un Nou Format JSON per a la Configuració del Model:**
    - El JSON actual defineix una llista de capes que s'apliquen seqüencialment.
    - El nou format haurà de permetre:
        - Identificar cada capa de manera única (p. ex., amb un camp `id`).
        - Especificar les entrades de cada capa. Aquestes entrades podran ser la sortida d'una o més capes anteriors, o l'entrada principal del model.
        - Definir l'ordre d'execució o com es resolen les dependències si no és estrictament seqüencial.

- [ ] **Modificació de la Lògica de Creació de Capes (`create_layers` o similar):**
    - En lloc de simplement crear una llista de capes per a `nn.Sequential`, aquesta funció haurà de parsejar el nou format JSON.
    - Probablement emmagatzemarà les capes en un `nn.ModuleDict` perquè es puguin referenciar per `id` durant el `forward pass`.

- [ ] **Reimplementació de la Classe `DynamicModel`:**
    - **`__init__`**: S'inicialitzaran les capes utilitzant la funció modificada del pas anterior.
    - **`forward`**: Aquesta serà la part més crítica. S'haurà de reescriure completament per:
        - Processar les capes segons les dependències definides en el JSON.
        - Guardar les sortides intermèdies de les capes per poder-les utilitzar com a entrades d'altres capes.
        - Gestionar la combinació de sortides si una capa rep múltiples entrades (p. ex., concatenació, suma). Aquesta lògica de combinació també podria ser part de la definició de la capa en el JSON.
        - Definir quina és la sortida final del model.

- [ ] **Creació d'un Nou Fitxer Python per al Model Builder Refactoritzat:**
    - Per mantenir l'original intacte, implementarem la nova lògica en un fitxer diferent.

- [ ] **Desenvolupament d'un Fitxer de Test:**
    - Crear un fitxer de test (`test_model_builder.py` o similar).
    - Definir diversos JSON d'exemple:
        - Un model seqüencial simple (per verificar la retrocompatibilitat o la capacitat de gestionar casos simples).
        - Un model amb una skip connection (similar a una ResNet block).
        - Un model amb una estructura més complexa (p. ex., una mini U-Net).
    - Els tests instanciaran els models a partir dels JSONs i, si és possible, faran una passada dummy per verificar que les dimensions i les connexions són correctes.

### Exemple de Nou Format JSON Proposat

```json
{
  "model_name": "SimpleResModel",
  "layers": [
    {
      "id": "input_data",
      "type": "Input"
    },
    {
      "id": "conv1",
      "type": "Conv2d",
      "params": {
        "in_channels": 3,
        "out_channels": 16,
        "kernel_size": 3,
        "padding": 1
      },
      "inputs": ["input_data"]
    },
    {
      "id": "relu1",
      "type": "ReLU",
      "params": {},
      "inputs": ["conv1"]
    },
    {
      "id": "conv2",
      "type": "Conv2d",
      "params": {
        "in_channels": 16,
        "out_channels": 16,
        "kernel_size": 3,
        "padding": 1
      },
      "inputs": ["relu1"]
    },
    {
      "id": "skip_connection_add",
      "type": "Add",
      "inputs": ["relu1", "conv2"]
    },
    {
      "id": "conv3",
      "type": "Conv2d",
      "params": {
        "in_channels": 16,
        "out_channels": 32,
        "kernel_size": 3,
        "padding": 1
      },
      "inputs": ["skip_connection_add"]
    }
  ],
  "output_layers": ["conv3"]
}
```

### Consideracions Addicionals sobre el JSON:

*   **Operacions de Combinació:** Per a capes que reben múltiples entrades (com `skip_connection_add`), necessitarem definir com es combinen aquestes entrades. Es podria fer amb capes "virtuals" o "d'operació" com `Add`, `Concat`, etc., que tindrien la seva pròpia lògica en el `forward pass`. El constructor hauria de saber com gestionar aquests tipus d'operacions.
*   **Ordre d'Execució:** El `forward pass` haurà de resoldre l'ordre correcte basant-se en les dependències d'`inputs`. Un ordenament topològic de les capes podria ser necessari si l'ordre en el JSON no garanteix que les entrades estiguin disponibles.
*   **Validació:** Seria bo afegir validació del JSON per assegurar que tots els `id` referenciats a `inputs` existeixen i que no hi ha cicles de dependència (tret que es vulguin models recurrents, la qual cosa afegeix més complexitat).

### Crítica Constructiva:

L'enfocament actual de `getattr(nn, layer_type)` és bo per la seva simplicitat, però a mesura que afegim operacions personalitzades (`Add`, `Concat`) o capes que no són directament de `torch.nn` (o que requereixen una inicialització més complexa basada en les seves entrades), haurem d'expandir aquesta lògica. Podríem tenir un diccionari de "constructors de capes/operacions" que mapegi els `type` del JSON a funcions que creen i retornen el mòdul o realitzen l'operació.

Aquest canvi és significatiu i afegeix una considerable complexitat, especialment en la implementació del mètode `forward`. No obstant això, la flexibilitat que s'aconsegueix és molt gran. 