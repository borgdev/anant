{{/*
Expand the name of the chart.
*/}}
{{- define "anant-enterprise.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "anant-enterprise.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "anant-enterprise.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "anant-enterprise.labels" -}}
helm.sh/chart: {{ include "anant-enterprise.chart" . }}
{{ include "anant-enterprise.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "anant-enterprise.selectorLabels" -}}
app.kubernetes.io/name: {{ include "anant-enterprise.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Ray Head labels
*/}}
{{- define "anant-enterprise.rayHeadLabels" -}}
{{ include "anant-enterprise.labels" . }}
app.kubernetes.io/component: ray-head
ray.io/node-type: head
{{- end }}

{{/*
Ray Worker labels
*/}}
{{- define "anant-enterprise.rayWorkerLabels" -}}
{{ include "anant-enterprise.labels" . }}
app.kubernetes.io/component: ray-worker
ray.io/node-type: worker
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "anant-enterprise.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "anant-enterprise.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Get the PostgreSQL secret name
*/}}
{{- define "anant-enterprise.postgresqlSecretName" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-%s" .Release.Name "postgresql" }}
{{- else }}
{{- printf "%s-%s" (include "anant-enterprise.fullname" .) "postgresql" }}
{{- end }}
{{- end }}

{{/*
Get the Redis secret name
*/}}
{{- define "anant-enterprise.redisSecretName" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-%s" .Release.Name "redis" }}
{{- else }}
{{- printf "%s-%s" (include "anant-enterprise.fullname" .) "redis" }}
{{- end }}
{{- end }}

{{/*
Generate JWT secret
*/}}
{{- define "anant-enterprise.jwtSecret" -}}
{{- if .Values.anant.security.jwtSecret }}
{{- .Values.anant.security.jwtSecret }}
{{- else }}
{{- randAlphaNum 32 }}
{{- end }}
{{- end }}

{{/*
Generate API key secret
*/}}
{{- define "anant-enterprise.apiKeySecret" -}}
{{- if .Values.anant.security.apiKeySecret }}
{{- .Values.anant.security.apiKeySecret }}
{{- else }}
{{- randAlphaNum 64 }}
{{- end }}
{{- end }}

{{/*
Generate Ray Redis password
*/}}
{{- define "anant-enterprise.rayRedisPassword" -}}
{{- if .Values.anant.ray.redisPassword }}
{{- .Values.anant.ray.redisPassword }}
{{- else }}
{{- randAlphaNum 16 }}
{{- end }}
{{- end }}

{{/*
Ray Head Service Name
*/}}
{{- define "anant-enterprise.rayHeadServiceName" -}}
{{- printf "%s-%s" (include "anant-enterprise.fullname" .) "ray-head" }}
{{- end }}

{{/*
Common environment variables
*/}}
{{- define "anant-enterprise.commonEnvVars" -}}
- name: RAY_DISABLE_IMPORT_WARNING
  value: "1"
- name: ANANT_MODE
  value: {{ .Values.anant.mode | quote }}
- name: ANANT_CLUSTER_NAME
  value: {{ .Values.anant.clusterName | quote }}
- name: ANANT_SECURITY_ENABLED
  value: {{ .Values.anant.security.enabled | quote }}
- name: ANANT_MONITORING_ENABLED
  value: {{ .Values.anant.enterprise.monitoring | quote }}
- name: PYTHONUNBUFFERED
  value: "1"
- name: PYTHONDONTWRITEBYTECODE
  value: "1"
{{- end }}

{{/*
Security environment variables
*/}}
{{- define "anant-enterprise.securityEnvVars" -}}
{{- if .Values.anant.security.enabled }}
- name: ANANT_JWT_SECRET
  valueFrom:
    secretKeyRef:
      name: {{ include "anant-enterprise.fullname" . }}-secrets
      key: jwt-secret
- name: ANANT_API_KEY_SECRET
  valueFrom:
    secretKeyRef:
      name: {{ include "anant-enterprise.fullname" . }}-secrets
      key: api-key-secret
{{- end }}
{{- end }}