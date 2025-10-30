---
TIPO: procesos_tesis
ENTIDAD: general
PALABRAS_CLAVE: proceso, etapas, proyecto, borrador, sustentación, director, jurados, coordinador
---

# Mapa de Funciones - Proceso de Tesis UNA Puno

## ACTORES_DEL_PROCESO
**TIPO:** actor_proceso
**ROL:** Tesista
**DESCRIPCION:** Estudiante de 7mo a 10mo semestre, egresado de la UNA Puno
**RESPONSABILIDADES:** Desarrollo de la investigación, entrega de documentos, sustentación

## ACTORES_DEL_PROCESO
**TIPO:** actor_proceso
**ROL:** Director de Tesis
**DESCRIPCION:** También conocido como Asesor de Tesis, es un docente ordinario de la misma escuela profesional
**RESPONSABILIDADES:** Revisión y aprobación del proyecto, orientación metodológica, seguimiento de la investigación

## ACTORES_DEL_PROCESO
**TIPO:** actor_proceso
**ROL:** Jurados
**DESCRIPCION:** Docentes ordinarios y contratados de la UNA Puno
**RESPONSABILIDADES:** Evaluación del proyecto y borrador, participación en la sustentación, asignación de calificación

## ACTORES_DEL_PROCESO
**TIPO:** actor_proceso
**ROL:** Sub Director Unidad de Investigación
**DESCRIPCION:** Docente ordinario de la escuela profesional
**RESPONSABILIDADES:** Supervisión del proceso a nivel de escuela profesional

## ACTORES_DEL_PROCESO
**TIPO:** actor_proceso
**ROL:** Director de Investigación de Facultad
**DESCRIPCION:** Resuelve conflictos a nivel facultad
**RESPONSABILIDADES:** Resolución de controversias, supervisión general del proceso

## ACTORES_DEL_PROCESO
**TIPO:** actor_proceso
**ROL:** Dirección de Instituto de Investigación
**DESCRIPCION:** Instancia superior para mediación
**RESPONSABILIDADES:** Mediación en conflictos complejos, supervisión institucional

## ACTORES_DEL_PROCESO
**TIPO:** actor_proceso
**ROL:** Sub Unidad de Plataforma
**DESCRIPCION:** Administra y desarrolla la plataforma
**RESPONSABILIDADES:** Gestión de la Plataforma de Gestión de la Investigación (PGI), soporte técnico

## ACTORES_DEL_PROCESO
**TIPO:** actor_proceso
**ROL:** Coordinador de Investigación
**DESCRIPCION:** Guía y supervisa todo el proceso
**RESPONSABILIDADES:** Revisión de formato, verificación de requisitos, coordinación general del proceso

## ETAPA_PROYECTO_TESIS
**TIPO:** etapa_proceso
**NOMBRE:** Proyecto de Tesis
**DESCRIPCION:** Primera etapa del proceso donde se presenta y aprueba el proyecto de investigación
**RESPONSABLE:** Tesista y Director de Tesis
**SUBPROCESOS:** Revisión del Director, Revisión del Coordinador
**RESULTADO_ESPERADO:** Proyecto aprobado para ejecución

## SUBETAPA_REVISION_DIRECTOR
**TIPO:** subetapa_proceso
**ETAPA_PADRE:** ETAPA_PROYECTO_TESIS
**NOMBRE:** Revisión por Director de Tesis
**DESCRIPCION:** El Director de Tesis revisa y aprueba el proyecto de investigación
**ACCIONES_APROBADO:** Firma digital en plataforma, autorización para continuar proceso
**ACCIONES_RECHAZADO:** Observaciones específicas, oportunidad de corrección y reenvío
**RECOMENDACION:** Programar reunión con Director para aclarar dudas antes de reenviar

## SUBETAPA_REVISION_COORDINADOR
**TIPO:** subetapa_proceso
**ETAPA_PADRE:** ETAPA_PROYECTO_TESIS
**NOMBRE:** Revisión por Coordinador de Investigación
**DESCRIPCION:** Revisión de aspectos formales del proyecto por el Coordinador de Investigación
**ASPECTOS_REVISADOS:** Formato según normativa universitaria, estructura del documento, márgenes y espaciado, numeración y orden de secciones, referencias bibliográficas
**ACCIONES_APROBADO:** Registro oficial del proyecto, autorización para iniciar investigación
**ACCIONES_RECHAZADO:** Lista detallada de correcciones, pausa del proceso hasta ajustes

## ETAPA_BORRADOR_TESIS
**TIPO:** etapa_proceso
**NOMBRE:** Borrador de Tesis
**DESCRIPCION:** Segunda etapa donde se entrega el borrador completo de la tesis para revisión formal
**RESPONSABLE:** Tesista y Coordinador de Investigación
**ASPECTOS_VERIFICADOS:** Paginación correcta, sistema de citas, tablas y figuras numeradas, índices actualizados, redacción y ortografía, formato de títulos, anexos completos
**RESULTADO_ESPERADO:** Borrador aprobado para proceder a sustentación

## PROCESO_BORRADOR_APROBADO
**TIPO:** flujo_proceso
**ETAPA:** ETAPA_BORRADOR_TESIS
**CONDICION:** Borrador cumple todos los requisitos
**ACCIONES:** Recepción de ejemplares corregidos, generación de documentación oficial, designación formal del jurado evaluador
**DOCUMENTOS_GENERADOS:** Memorándum para convocar Reunión de Dictamen, Oficios de notificación a jurados

## PROCESO_BORRADOR_RECHAZADO
**TIPO:** flujo_proceso
**ETAPA:** ETAPA_BORRADOR_TESIS
**CONDICION:** Borrador no cumple requisitos
**ACCIONES:** Devolución con observaciones puntuales, tiempo para correcciones, pausa del proceso hasta subsanación
**RECOMENDACION:** Solicitar revisión previa con Director antes de entrega oficial

## ETAPA_SUSTENTACION
**TIPO:** etapa_proceso
**NOMBRE:** Sustentación de Tesis
**DESCRIPCION:** Tercera etapa donde se presenta y defiende la tesis ante el jurado evaluador
**RESPONSABLE:** Tesista, Jurados, Coordinador de Investigación
**INFORMACION_REQUERIDA:** Título final, resumen ejecutivo, palabras clave, datos de miembros del jurado, fecha y hora propuestas, lugar de sustentación
**RESULTADO_ESPERADO:** Tesis sustentada exitosamente y calificada

## SUBETAPA_REGISTRO_INFORMACION
**TIPO:** subetapa_proceso
**ETAPA_PADRE:** ETAPA_SUSTENTACION
**NOMBRE:** Registro de Información Final
**DESCRIPCION:** Ingreso de información definitiva de la tesis al sistema antes de la sustentación
**DATOS_REQUERIDOS:** Título final, resumen ejecutivo, palabras clave, datos del jurado, fecha y hora, lugar
**VALIDACION:** Coordinador verifica que toda la información esté correcta

## SUBETAPA_VALIDACION_COORDINADOR
**TIPO:** subetapa_proceso
**ETAPA_PADRE:** ETAPA_SUSTENTACION
**NOMBRE:** Validación por Coordinador
**DESCRIPCION:** Verificación final de que toda la información esté en orden antes de la sustentación
**ACCIONES_APROBADO:** Publicación del Comunicado Oficial de Sustentación, difusión por portal web, correos institucionales, redes sociales, vitrinas informativas
**ACCIONES_RECHAZADO:** Notificación de errores, requerimiento de correcciones, retraso del proceso

## EVENTO_SUSTENTACION
**TIPO:** evento_proceso
**ETAPA:** ETAPA_SUSTENTACION
**NOMBRE:** Día de la Sustentación
**DESCRIPCION:** Acto académico donde se presenta y defiende la tesis
**PREPARACION:** Llegar 30 minutos antes, probar presentación y equipo
**DESARROLLO:** Exposición del trabajo (20-30 minutos), ronda de preguntas del jurado, deliberación del jurado, veredicto y calificación
**RESULTADO_EXITOSO:** Tesis sustentada, calificación oficial recibida, inicio de trámite de titulación

## CONSEJOS_PROCESO
**TIPO:** consejo_proceso
**CATEGORIA:** comunicación
**DESCRIPCION:** Mantener comunicación constante con el Director de Tesis

## CONSEJOS_PROCESO
**TIPO:** consejo_proceso
**CATEGORIA:** documentación
**DESCRIPCION:** Guardar todas las versiones del trabajo

## CONSEJOS_PROCESO
**TIPO:** consejo_proceso
**CATEGORIA:** plazos
**DESCRIPCION:** Cumplir los plazos establecidos por la facultad

## CONSEJOS_PROCESO
**TIPO:** consejo_proceso
**CATEGORIA:** formato
**DESCRIPCION:** No dejar para último momento las correcciones de formato

## CONSEJOS_PROCESO
**TIPO:** consejo_proceso
**CATEGORIA:** preparación
**DESCRIPCION:** Practicar la sustentación al menos 5 veces antes del día D

## CONSEJOS_PROCESO
**TIPO:** consejo_proceso
**CATEGORIA:** preparación
**DESCRIPCION:** Anticipar preguntas difíciles y preparar respuestas

## CONSEJOS_PROCESO
**TIPO:** consejo_proceso
**CATEGORIA:** actitud
**DESCRIPCION:** Mantener la calma - cientos lo han logrado antes

## TIEMPOS_PROCESO
**TIPO:** tiempo_proceso
**ETAPA:** Aprobación del Proyecto
**TIEMPO_ESTIMADO:** 2-4 semanas

## TIEMPOS_PROCESO
**TIPO:** tiempo_proceso
**ETAPA:** Desarrollo de la Investigación
**TIEMPO_ESTIMADO:** 6-12 meses

## TIEMPOS_PROCESO
**TIPO:** tiempo_proceso
**ETAPA:** Revisión del Borrador
**TIEMPO_ESTIMADO:** 2-3 semanas

## TIEMPOS_PROCESO
**TIPO:** tiempo_proceso
**ETAPA:** Preparación para Sustentación
**TIEMPO_ESTIMADO:** 2-4 semanas

## TIEMPOS_PROCESO
**TIPO:** tiempo_proceso
**ETAPA:** TOTAL
**TIEMPO_ESTIMADO:** 8-15 meses
**NOTA:** Los tiempos varían según la facultad y la complejidad de la investigación