import axios from 'axios';

const BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? 'http://127.0.0.1:8000/api/v1' : '/api/v1');

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

export interface ApiError {
  message: string;
  status?: number;
  code?: string;
  data?: any;
}

// Response interceptor
apiClient.interceptors.response.use(
  (response) => {
    // Return the envelope data directly
    return response.data;
  },
  (error) => {
    let message = 'An unexpected error occurred';
    const status = error.response?.status;
    const code = error.code;

    if (error.code === 'ECONNABORTED') {
      message =
        'Request timed out. Please check if the local backend server is running and try again.';
    } else if (error.message === 'Network Error' || !error.response) {
      message = 'Could not connect to the ChitraAI service. Please try again shortly.';
    } else if (error.response?.data?.detail) {
      if (typeof error.response.data.detail === 'string') {
        message = error.response.data.detail;
      } else if (Array.isArray(error.response.data.detail)) {
        // Parse validation errors
        message = error.response.data.detail
          .map((d: any) => d.msg || d.message || JSON.stringify(d))
          .join(', ');
      }
    } else if (error.response?.data?.message) {
      message = error.response.data.message;
    }

    const errorDetails: ApiError = {
      message,
      status,
      code,
      data: error.response?.data,
    };

    return Promise.reject(errorDetails);
  }
);

export default apiClient;
